# Handwritten Digits Classification (MNIST)

Train a simple CNN on the MNIST handwritten digits dataset and run inference either:
- in the notebook (test-set prediction + plot), or
- via an OpenCV drawing canvas that predicts the digit you draw.

## Downloads (Google Drive)

Some large files are not uploaded to this GitHub repository (model / data / demo video). Download them from Google Drive:

- Model weights (`digit_cnn.pth`): [link](https://drive.google.com/file/d/1_5ocbHc0au0V4cWIeEZbsOZzBHZoKZN4/view?usp=sharing)
- Demo video (optional): [link](https://drive.google.com/file/d/16m1JzbwQ5ySYeCuTxKr8SUlaLTi6M28r/view?usp=sharing)
- After downloading the model, place `digit_cnn.pth` in the project root (same folder as `OnCamaraDetact.py`).

## Project Structure

- `main.ipynb` — training + evaluation + saving `digit_cnn.pth`
- `digit_cnn.pth` — saved PyTorch model weights (`state_dict`) (download from Drive, or generate by training)
- `OnCamaraDetact.py` — interactive OpenCV canvas to draw a digit and predict it
- `data/` — MNIST dataset download/cache location (created locally by the notebook)

## Requirements

- Python 3.x
- Packages:
  - `torch`
  - `torchvision`
  - `matplotlib` (used in the notebook)
  - `opencv-python` (used by `OnCamaraDetact.py` and `check.py`)
  - `numpy`

Install (recommended in a venv):

```bash
pip install torch torchvision matplotlib opencv-python numpy
```

## Train & Save the Model

Training is in `main.ipynb`:

- Downloads MNIST to `data/`
- Trains for `epochs = 5`
- Prints epoch loss
- Evaluates accuracy on the test set
- Saves weights to `digit_cnn.pth`

Open and run the notebook:

```bash
jupyter notebook main.ipynb
```

After running, you should have:

- `digit_cnn.pth` in the project root

## Run: Draw a Digit and Predict

`OnCamaraDetact.py` opens a window with a blank canvas.

Controls:
- Draw with **left mouse button**
- Press `p` to predict
- Press `c` to clear
- Press `ESC` to exit

Run:

```bash
python OnCamaraDetact.py
```

Notes:
- The script loads `digit_cnn.pth` and runs the model on CPU or GPU automatically.
- The input is resized to 28×28 and normalized the same way as training.

## Troubleshooting

- **`FileNotFoundError: digit_cnn.pth`**
  - Download the model from the Google Drive link above, or run the training notebook to generate it.
  - Ensure `digit_cnn.pth` is in the same directory as `OnCamaraDetact.py`.

- **OpenCV window doesn’t open / crashes on Linux**
  - Make sure you installed `opencv-python` (not just `opencv`).
  - If you are on a headless environment (no GUI), OpenCV windows won’t display.

- **`torch.load(..., weights_only=True)` error**
  - Some PyTorch versions may not support the `weights_only` argument.
  - If you hit an error, remove `weights_only=True` from the `torch.load(...)` call in `OnCamaraDetact.py`.

## License


- Choose a license: https://choosealicense.com/
- After you add a `LICENSE` file to this repo, it will be available at: `LICENSE`
