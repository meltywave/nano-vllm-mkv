import socket
import torch
import io
import time
import logging

# 模块独立日志，彻底消除和nanovllm包的循环导入依赖
logger = logging.getLogger("remote_kv_client")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class RemoteKVClient:
    def __init__(self, host="127.0.0.1", port=12345, connect_retry=5, op_retry=3, timeout=10.0):
        """
        新增参数：
        - op_retry: 单次读写操作的重试次数（网络波动时自动重试）
        - timeout: 单次socket操作超时时间（秒）
        """
        self.host = host
        self.port = port
        self.sock = None
        self.connect_retry = connect_retry
        self.op_retry = op_retry
        self.timeout = timeout
        
        # 统计指标（字节数、操作次数）
        self.stats = {
            "total_send_bytes": 0,
            "total_recv_bytes": 0,
            "send_success": 0,
            "recv_success": 0,
            "retry_count": 0,
            "fail_count": 0,
            # 累计耗时统计（毫秒）
            "total_upload_ms": 0.0,
            "total_download_ms": 0.0
        }
        
        self._connect()

    def _connect(self):
        """连接服务端，失败重试connect_retry次"""
        for attempt in range(self.connect_retry):
            try:
                # 先关闭旧连接（如果有）
                if self.sock:
                    try:
                        self.sock.close()
                    except:
                        pass
                
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect((self.host, self.port))
                logger.info(f"[Client] Connect success {self.host}:{self.port}")
                return True
            except Exception as e:
                logger.warning(f"[Client] Connect failed attempt {attempt+1}/{self.connect_retry}, err:{e}")
                time.sleep(1.0)
        
        # 连接彻底失败，不再抛出异常，改为标记不可用，实现降级
        logger.warning(f"[Client WARN] 无法连接远端服务 {self.host}:{self.port}，远端Swap功能将自动降级")
        self.sock = None
        return False

    def _is_available(self):
        """检查客户端是否可用（连接正常）"""
        return self.sock is not None

    def _recv_exact(self, length):
        buffer = b""
        while len(buffer) < length:
            chunk = self.sock.recv(length - len(buffer))
            if not chunk:
                raise ConnectionError("socket closed by peer")
            buffer += chunk
        return buffer

    def _send_packet(self, data_bytes: bytes):
        length_header = len(data_bytes).to_bytes(8, byteorder="big")
        self.sock.sendall(length_header)
        self.sock.sendall(data_bytes)
        # 统计发送字节数（包含8字节长度头）
        self.stats["total_send_bytes"] += len(data_bytes) + 8

    def _recv_packet(self):
        header = self._recv_exact(8)
        data_len = int.from_bytes(header, byteorder="big")
        payload = self._recv_exact(data_len)
        # 统计接收字节数（包含8字节长度头）
        self.stats["total_recv_bytes"] += data_len + 8
        return payload

    def _reconnect_on_error(self):
        """网络出错时尝试重连一次"""
        logger.info("[Client] 检测到连接异常，尝试重新连接...")
        return self._connect()

    def put_block(self, block_id: int, tensor: torch.Tensor) -> bool:
        """
        上传块到远端，支持重试和自动重连
        单块张量大小限制（默认128MB），超大块直接跳过保护带宽
        返回：True成功，False失败（上层可根据返回值做降级）
        """
        start_time = time.time()

        # 先检查客户端是否可用
        if not self._is_available():
            self.stats["fail_count"] += 1
            return False

        # 张量大小校验，超过128MB直接拒绝上传，防止超大KV块占满网络
        tensor_mb = tensor.nbytes / 1024 / 1024
        MAX_BLOCK_MB = 128
        if tensor_mb > MAX_BLOCK_MB:
            logger.warning(f"[Client WARN] put_block {block_id} 张量大小 {tensor_mb:.2f}MB 超过限制 {MAX_BLOCK_MB}MB，跳过上传")
            self.stats["fail_count"] += 1
            return False

        last_error = None
        for attempt in range(self.op_retry):
            try:
                buff = io.BytesIO()
                torch.save({
                    "cmd": "PUT",
                    "block_id": block_id,
                    "tensor": tensor
                }, buff)
                raw = buff.getvalue()
                
                self._send_packet(raw)
                resp_bytes = self._recv_packet()
                
                if resp_bytes == b"OK":
                    # 统计上传耗时
                    cost_ms = (time.time() - start_time) * 1000
                    self.stats["send_success"] += 1
                    self.stats["total_upload_ms"] += cost_ms
                    logger.info(f"[Client INFO] 上传块 {block_id} 成功 | 大小 {tensor_mb:.2f}MB | 耗时 {cost_ms:.2f}ms")
                    return True
                else:
                    last_error = f"server returned {resp_bytes}"
                    logger.error(f"[Client ERROR] put_block {block_id} server error: {last_error}")
                    
            except (ConnectionError, socket.timeout, BrokenPipeError) as e:
                # 网络类错误，尝试重连后再重试
                last_error = str(e)
                logger.warning(f"[Client WARN] put_block {block_id} network error attempt {attempt+1}/{self.op_retry}: {e}")
                self.stats["retry_count"] += 1
                # 重连一次
                self._reconnect_on_error()
                time.sleep(0.3)
                
            except Exception as e:
                # 其他未知错误，直接失败
                last_error = str(e)
                logger.error(f"[Client ERROR] put_block {block_id} unexpected error: {e}")
                break

        # 所有重试都失败
        self.stats["fail_count"] += 1
        logger.error(f"[Client ERROR] put_block {block_id} 最终失败，跳过该块驱逐")
        return False

    def get_block(self, block_id: int) -> torch.Tensor | None:
        """
        从远端获取块，支持重试和自动重连
        返回：成功返回tensor，失败返回None（上层需处理None的情况，比如重新计算）
        失败不抛出异常，避免程序崩溃，实现降级
        """
        start_time = time.time()

        if not self._is_available():
            self.stats["fail_count"] += 1
            logger.error(f"[Client ERROR] get_block {block_id} 客户端不可用，无法获取远端块")
            return None

        last_error = None
        for attempt in range(self.op_retry):
            try:
                buff = io.BytesIO()
                torch.save({"cmd": "GET", "block_id": block_id}, buff)
                raw = buff.getvalue()
                
                self._send_packet(raw)
                payload = self._recv_packet()
                
                if payload == b"ERR":
                    # 远端明确说没有这个块，不需要重试
                    logger.error(f"[Client ERROR] get_block {block_id} 远端不存在该块")
                    self.stats["fail_count"] += 1
                    return None
                
                res = torch.load(io.BytesIO(payload), weights_only=False)
                tensor = res["tensor"]
                # 统计下载耗时
                cost_ms = (time.time() - start_time) * 1000
                tensor_mb = tensor.nbytes / 1024 / 1024
                self.stats["recv_success"] += 1
                self.stats["total_download_ms"] += cost_ms
                logger.info(f"[Client INFO] 下载块 {block_id} 成功 | 大小 {tensor_mb:.2f}MB | 耗时 {cost_ms:.2f}ms")
                return tensor
                
            except (ConnectionError, socket.timeout, BrokenPipeError) as e:
                last_error = str(e)
                logger.warning(f"[Client WARN] get_block {block_id} network error attempt {attempt+1}/{self.op_retry}: {e}")
                self.stats["retry_count"] += 1
                self._reconnect_on_error()
                time.sleep(0.3)
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"[Client ERROR] get_block {block_id} unexpected error: {e}")
                break

        self.stats["fail_count"] += 1
        logger.error(f"[Client ERROR] get_block {block_id} 最终失败: {last_error}")
        return None

    def get_stats(self):
        """获取客户端统计数据"""
        return self.stats.copy()

    def reset_stats(self):
        """重置统计数据"""
        for k in self.stats:
            self.stats[k] = 0