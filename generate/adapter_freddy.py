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
INPUT = """Articles:    # of articles here: 3 \n title: FW-WIFI User Guide (Freshworks Production WIFI)\n context:   \n FW-WIFI User Guide (Freshworks Production WIFI) \n Overview \n To enable seamless WiFi connectivity across all Geo-location offices and to strengthen our WiFi-related security posture, we are transitioning to a certificate-based WiFi authentication wherein the requirement to input a manual Password is eliminated. \xa0Effectively, that means we are decommissioning our old Wifi\xa0‘FDWIFI’\xa0which only supports “Password” as the authentication mechanism, and moves towards a stronger certificate-based authentication mechanism.\xa0 \n Our new WiFi (SSID Name:\xa0FW-WIFI) is certificate-based. It can prevent over-the-air credential theft attacks like Man-in-the-Middle attacks and Evil Twin proxies. It is much more secure than Pre-Shared Key networks, which are typically used in personal networks. \n Certificate-based authentication significantly reduces an organization’s risk for credential theft. Not only does it stop credentials from being sent over the air where they can be easily stolen, but it forces users to go through an enrollment/onboarding process that ensures their devices are configured correctly. \n How to Connect to the FW-WIFI \n   On your laptop, list the available WiFi networks. WiFi (SSID):\xa0’FW-WIFI’, will be among the various listed. Please click on it and it will connect automatically on every production laptop.\xa0   \n How it works \n \n  To connect to the FW-WIFI, the laptops need a valid user certificate. IT will centrally push the required certificates and profiles to all end-user laptops/Macbooks. It is a silent installation in the background.\xa0  \n  All employee Laptops/MacBooks will need to be Azure joined to be able to receive the certificates.\xa0Any machines that are not Azure joined, will not have certificates and will not be able to access the WiFi Network.\xa0  \n  Other OS laptops (Linux) need to be enrolled with the help of the IT Team  \n  During authentication, the certificate and AD membership are validated to verify the right to access the WiFi network.  \n  After successful validation, based on the user’s department, the user will get access to the internet and appropriate network resources.\xa0  \n \n   \n FW-WiFi Connectivity Procedure \n Listed below are the steps to connect to the new WiFi \n Windows Laptops: \n \n  Click on the WiFi icon which you can find on\xa0the taskbar.\xa0  \n  Verify that WiFi is turned on, this will then list the available WiFi networks.  \n  Click on “FW-WIF“ to connect WiFi.  \n \n   \n \xa0 \n   \n Apple MacBook \n \n  Click on the WiFi icon which you can find on the\xa0menu bar.\xa0  \n  Verify that Wifi is turned on, this will then list the available WiFi networks.  \n  Click on the SSID “ ‘FW-WIFI “ to connect to the new WiFi network.  \n \n  \n Connectivity for mobiles, other personal handheld devices, and guest devices.\xa0 \n \n  The SSID for Employee personal devices is\xa0FW-MOBILE.\xa0Go to\xa0https://confluence.freshworks.com/x/T7qyEw\xa0for the\xa0FW-MOBILE\xa0 user guide  \n  The SSID for guest devices is\xa0FW-VISITORS. Guest pass token is required to connect to this SSID; please contact the IT Team to get the guest pass token.\xa0  \n \n FAQs: \n \n  What will I do if I am not able to connect to the WiFi?  \n  Users will not be able to connect to the WiFi if the profile & policies are not synced properly. If the connectivity is not established, please contact\xa0the IT Team\xa0via\xa0email\xa0(help@freshworks.com) or via Slack channel #global-help. We will assist you promptly.  \n \n   \n   If there is a requirement for connecting your testing devices or other OS devices (Linux, Android, etc) to the FW-WIFI, please contact the IT Team \xa0via\xa0email\xa0(help@freshworks.com) or via Slack channel #global-help\xa0to enable these devices to use FW-WIFI   \n   \n   Our current production SSID ‘FDWIFI’ will run in parallel for a while and then be decommissioned. We will send out a communication prior to the decommissioning.   \n   \n   \n   \n   \n title: How to Check Zoom Connection Issue\n context:  If you are experiencing any issue(s) with latency, frozen screen, poor quality audio, or meeting getting disconnected while using a home or non-enterprise WiFi connection \n   \n   \n  \n   \n Check the following steps to solve the issues. \n   \n \n Check the Zoom version is up-to date upgraded. \n Quit the Zoom app and reopen again. \n Check Home internet wifi connectivity. \n Try to connect directly via Wired (if your internet router has wired ports) \n Try bringing your computer or mobile device closer to the WiFi router or access point in your home or office. \n Check the Global Protect(SWG) Connection is connected. \n \n   \n If you have trouble completing the above steps or if there are issues after following them, please raise an incident ticket at\xa0https://lighthouse.freshservice.com/support/tickets/new\xa0or send a mail to IT via\xa0help@freshworks.com\xa0or post a message on the\xa0#global-help\xa0Slack channel. \n User Query: wifi issues"""
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
