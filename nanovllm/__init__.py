__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
    if name == "LLM":
        from nanovllm.llm import LLM

        return LLM
    if name == "SamplingParams":
        from nanovllm.sampling_params import SamplingParams

        return SamplingParams
    raise AttributeError(name)
