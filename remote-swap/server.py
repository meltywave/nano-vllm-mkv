# remote_swap/server.py
import socket
import torch
import io
import threading

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 12345
kv_storage = {}

def recv_exact(sock, length):
    buffer = b""
    while len(buffer) < length:
        data = sock.recv(length - len(buffer))
        if not data:
            return None
        buffer += data
    return buffer

def send_packet(sock, payload: bytes):
    header = len(payload).to_bytes(8, byteorder="big")
    sock.sendall(header)
    sock.sendall(payload)

def handle_client(conn):
    global kv_storage
    conn.settimeout(15.0)
    try:
        with conn:
            while True:
                header = recv_exact(conn, 8)
                if header is None:
                    break
                pkg_len = int.from_bytes(header, byteorder="big")
                payload_bytes = recv_exact(conn, pkg_len)
                if payload_bytes is None:
                    break
                buff = io.BytesIO(payload_bytes)
                data = torch.load(buff, weights_only=False)
                cmd = data["cmd"]
                block_id = data["block_id"]

                if cmd == "PUT":
                    kv_storage[block_id] = data["tensor"]
                    send_packet(conn, b"OK")
                elif cmd == "GET":
                    if block_id in kv_storage:
                        send_buff = io.BytesIO()
                        torch.save({"tensor": kv_storage[block_id]}, send_buff)
                        send_data = send_buff.getvalue()
                        send_packet(conn, send_data)
                    else:
                        send_packet(conn, b"ERR")
    except Exception as e:
        print(f"[Server] Client connection exception: {e}")

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((SERVER_HOST, SERVER_PORT))
    except OSError:
        print(f"[Server] Port {SERVER_PORT} already in use!")
        return
    sock.listen(4)
    print(f"Remote KV Server running on {SERVER_HOST}:{SERVER_PORT}")
    while True:
        try:
            client_conn, addr = sock.accept()
            t = threading.Thread(target=handle_client, args=(client_conn,))
            t.daemon = True
            t.start()
        except Exception as e:
            print(f"[Server] Accept error: {e}")

if __name__ == "__main__":
    main()