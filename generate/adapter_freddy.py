import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama import Tokenizer
from lit_llama.adapter import LLaMA
from lit_llama.utils import EmptyInitOnDevice, lazy_load, llama_model_lookup
from scripts.prepare_alpaca import generate_prompt


PROMPT = """Operating as a virtual IT assistant, address user questions utilizing available articles or relying on comprehensive expertise"""
INPUT = """Articles:   # of articles here: 2 \n title: Laptop Policy\n context:  Objective: \n Freshworks issues laptop devices to employees as part of their on-boarding process to perform their job function. The objective of this document is to describe the policy for the laptop that is provided to the employees \n   \n Policy: \n 1. The laptop that is given to the employee when they start with Freshworks is for them to keep. Employees are expected to maintain it like the way they would maintain their personal items.\xa0 \n a) If the employee has had the same device during a 4-year period as an employee of Freshworks, then they get to keep the device irrespective of their employment status.\xa0 \n b) IT will refresh the laptops that are over four years old in a phased manner. When the new laptop is issued to the employee, the old laptop needs to be sanitized by IT before the employee gets to keep it for personal use.\xa0 \n c) As an example, if an employee has worked at Freshworks for five years, ad leaves the business, they would have received a laptop (Laptop-1) at the time of on-boarding, and another laptop (Laptop-2) after four years as part of the laptop refresh process. In this example, the employee gets to keep Laptop-1 (after it has been sanitized), since they have served for more than four years. Laptop-2 must be returned to Freshworks IT as part of the employee separation process.  \n \n \xa0\n\xa0\xa0 \n  \n 2. If the employee breaks the laptop, or spills fluids, or performs an action that renders             the laptop unusable, or causes to have it stolen carelessly within the first four years:\xa0 \n a) The 4-year clock for owning the laptop is reset every time a repair/replace is done. \n \n b) The employee will be given a laptop or a desktop, which may not be as powerful as the laptop they originally were issued, for a couple of months while their laptop is being repaired/replaced\xa0 \n c) In the event of a theft of the laptop, the employee will be responsible for paying 50% of the value of the laptop as a penalty. In the event of a theft, the employee is responsible for filing a police complaint, and presenting a copy of the police complaint to Freshworks IT. If the employee does not present a police complaint, they are responsible for paying 100% of the value of the laptop.\xa0 \n d) IT will make exceptions on a case by case basis, subject to the approval of management.\xa0 \xa0  \n The idea of this clause is NOT to penalize the employee but to act as a deterrent for careless use and handling of the laptop.  \n \n   \n \n 3. If the employee leaves the business before the completion of four years, they will be           required to return the laptop to IT.\xa0 \n   \n 4. This policy goes into effect for all new employees from the policy effective date           indicated above. For all existing employees, the policy will be rolled out in a phased           manner.\xa0 \n   \n 5. In case of ambiguity, IT reserves the right to interpret the policy. \n User Query: Unable to switch on my laptop"""
def main(
    prompt: str = PROMPT,
    input: str = INPUT,
    adapter_path: Path = Path("out/adapter/alpaca/lit-llama-adapter-finetuned.pth"),
    pretrained_path: Path = Path("checkpoints/lit-llama/7B/lit-llama.pth"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    quantize: Optional[str] = None,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned LLaMA-Adapter model.
    See `finetune_adapter.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        adapter_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune_adapter.py`.
        input: Optional input (Alpaca style).
        pretrained_path: The path to the checkpoint with pretrained LLaMA weights.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
    """
    assert adapter_path.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()

    fabric = L.Fabric(devices=1)
    dtype = torch.bfloat16 if fabric.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    with lazy_load(pretrained_path) as pretrained_checkpoint, lazy_load(adapter_path) as adapter_checkpoint:
        name = llama_model_lookup(pretrained_checkpoint)

        with EmptyInitOnDevice(
                device=fabric.device, dtype=dtype, quantization_mode=quantize
        ):
            model = LLaMA.from_name(name)

        # 1. Load the pretrained weights
        model.load_state_dict(pretrained_checkpoint, strict=False)
        # 2. Load the fine-tuned adapter weights
        model.load_state_dict(adapter_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup_module(model)

    tokenizer = Tokenizer(tokenizer_path)
    sample = {"instruction": prompt, "input": input}
    prompt = generate_prompt(sample)
    print("Final Input: ",prompt)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
    prompt_length = encoded.size(0)

    t0 = time.perf_counter()
    y = generate(model, encoded, max_new_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
    t = time.perf_counter() - t0

    model.reset_cache()
    output = tokenizer.decode(y)
    output = output.split("### Response:")[1].strip()
    print(output)

    tokens_generated = y.size(0) - prompt_length
    print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
    if fabric.device.type == "cuda":
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    CLI(main)
