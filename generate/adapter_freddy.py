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


PROMPT = """Hello! I will assist you with your query. If your query is related to any software tool, app, or falls under the domains of IT, Software, Networking, HR, Legal, Payroll, Finance, and Tax, I would be happy to help you.\n\nI will never ask any follow-up questions. I will look for answers in my secret library of solution articles that only I have access to. If I find an article that can help, I will extract a descriptive answer to your query based on the article\'s content but will never point you to the articles since you can\'t access them. If I am unable to find an answer in our articles or require additional information, I will provide a solution based on common knowledge and mention clearly as a "Concluding Note" that it\'s not from the article.\n\nIf your query is vague, irrelevant, inconclusive, or gibberish, I will provide an apologetic reply and ask you to re-write your entire query with more details."""
INPUT = "User Query: water spilled on my laptop  \n Articles:   # of articles here: 3 \n title: Update profile picture\n context:  People can see your profile photo on your public information page, in the directory search results, and the global header. You can have only one profile photo at a time. Click here\xa0https://mindtickle.app.link/zUQwGh4ngnb\xa0or follow the below steps: \n  1. Go to Me > Quick Actions > More, and then select Change Photo action.  Note:\xa0You may also update your photo using the My Photo page in general preferences. Click your user image or name in the global header and go to Personalization > Set Preferences > My Photo.  2. Click Choose File and select the photo to upload. Keep these points in mind when selecting the photo: - The file size should be less than 20 megabytes. - The preferable file format is .png or .jpeg though other image file formats are also supported. - Ensure that the image dimension is 90 x 120 pixels to avoid distortion. If the image isn't of this dimension, try maintaining an aspect ratio of 3 x 4.  3. Click Save and Close. \n   \n title: Make corrections to data\n context:  In case you have made any incorrect changes in PeopleHub, please try to edit and submit the correct information again. If this doesn't help, please raise a ticket in Lighthouse and the team would be able to help you on this. \n title: Where can I see my Total Compensation Plan?\n context:  The complete breakup of your Total Compensation Plan (Fixed, Variable, Equity) can be checked in PeopleHub under My Compensation.\xa0 \n   \n Here you can view your compensation details, such as salary and personal contributions. \n   \n Click\xa0here to view the video\xa0or follow the steps below.\xa0 \n   \n Go to Me > Apps > Personal Info >\xa0My Compensation\xa0 \n"
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
