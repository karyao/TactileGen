# Run Gemma on a remote machine

This example copies a chart from a Mac to the remote TactileGen project and
opens the Gemma interface through an SSH tunnel. Replace the username, host,
port, paths, or environment name if yours differ.

## 1. Copy the image from the Mac

In a terminal on the Mac, run:

```bash
scp -P 24 ~/Desktop/download.png kjy6@asb9700u-a06.csil.sfu.ca:~/Desktop/TactileGen/
```

The uppercase `-P` specifies the SSH port; lowercase `-p` preserves file
times and permissions. To check the copied file after connecting, run
`ls -l ~/Desktop/TactileGen/download.png` on the remote machine.

## 2. Prepare and start the app on the remote machine

Connect from the Mac, then activate the existing virtual environment:

```bash
ssh -p 24 kjy6@asb9700u-a06.csil.sfu.ca
cd ~/Desktop/TactileGen
source .venv/bin/activate
```

If the environment has another name, use that name in place of `.venv`.
Install the app and Gemma dependencies together so pip resolves compatible
versions:

```bash
python -m pip install -r requirements-vlm-app.txt
```

For the default gated Gemma checkpoint, accept its
[Hugging Face model terms](https://huggingface.co/google/gemma-3-4b-it) and
authenticate on the remote machine with `hf auth login`. The first extraction
may download the model and take several minutes.

Start the app:

```bash
python -m src.gradio_app --workflow gemma --gemma-dtype bfloat16
```

Leave this terminal running. This setting worked on the example machine and
requires a CUDA GPU with native bfloat16 support.

## 3. Open the app on the Mac

In a **new terminal on the Mac**, create the tunnel:

```bash
ssh -p 24 -L 7860:127.0.0.1:7860 kjy6@asb9700u-a06.csil.sfu.ca
```

Leave this connection open and visit <http://127.0.0.1:7860> in the Mac's
browser. Select the Gemma tab, upload the chart image, and select **Extract
chart**. Review the extracted text and values before generating tactile pages.
The browser upload uses the image on the Mac; the copied remote file can also
be used with the [command-line extraction workflow](gemma_baseline.md#command-line-extraction).

## Troubleshooting

- **Connection refused:** Check the hostname and port and whether network or
  VPN access is required. `scp` uses `-P 24`; `ssh` uses `-p 24`.
- **Missing app dependencies:** Activate the intended environment and install
  `requirements-vlm-app.txt` as shown above, then restart the app.
- **Out of GPU memory:** Float32 needs more memory than bfloat16. Stop other
  GPU workloads if appropriate, or use a machine with sufficient memory.
- **Unsupported bfloat16:** This option requires native CUDA bfloat16 support;
  use compatible hardware or a configuration with enough memory for float32.
- **Parsing error or empty output:** Open **Gemma diagnostics** and inspect
  **Raw model response** and **Extraction metadata**. Confirm
  `requested_dtype` and `resolved_dtype` are `bfloat16`. If
  `output_budget_reached` is true, the model hit its token limit; this alone
  does not explain an empty response. Save the metadata and any response for
  troubleshooting. The CLI can also save a decoded response and metadata even
  on parsing failure; see the linked baseline guide.

Stop the app with `Ctrl+C` in its remote terminal. Close the tunnel by exiting
the separate SSH session.
