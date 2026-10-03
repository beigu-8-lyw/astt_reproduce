# Transformer-based Spatio-Temporal Unsupervised Traffic Anomaly Detection in Aerial Videos

Authors: Tung Minh Tran, Doanh C. Bui, Tam V. Nguyen, Khang Nguyen

> Anomaly detection is an area of video analysis and plays an increasing role in ensuring safety, preventing risks, and guaranteeing quick response in intelligent surveillance systems. It has become popular research topics and piqued the interest of researchers in different communities, such as computer vision, machine learning, remote sensing, and data mining in recent years. This promotes novel mobile systems where drones are equipped with cameras to help people find better and more efficient solutions to automatically detect anomalies (e.g., car accidents, traffic congestion, street fighting) in traffic surveillance videos. However, anomaly detection methods are still rarely studied and developed in the remote sensing community due to anomalous events rarely occurring in real life, along with the high similarities between the objects of interest with small sizes, multi-scale objects, complex backgrounds of great variations, and high overlap between objects. Therefore, in order to fully exploit the spatio-temporal information for anomaly detection in traffic surveillance circumstances, we propose a future frame prediction network based on transformer architectures to detect abnormal events from drone videography in an unsupervised way. Our model treats consecutive video frames from an input clip and feeds features to a transformer encoder to capture spatial and temporal representations from the sequence, and then leverages a decoder to predict the next frame. Furthermore, an event is identified with high reconstruction error as an anomaly in the test phase. Thoroughly empirical studies demonstrate that our method achieves superior performance on the UIT-ADrone dataset and largely outperforms the state-of-the-art anomaly methods on the Drone-Anomaly dataset in aerial surveillance.

![overview](https://github.com/Tungufm/ASTT/assets/56221762/4040f597-d266-4e55-9505-8d47440141eb)

## Progress

We aim to update following documents for our repo, will finish soon:

- [ ] Training document
- [ ] Inference document
- [ ] Checkpoints
- [ ] Dataset description

## Training
> UIT-ADrone dataset includes 206,194 video frames. There are 59,186 frames for the training set and 147,005 frames for the test set. Notably, the training set only includes normal samples, whereas the test set consists of normal and abnormal patterns.

### Paper-derived full model implementation

This checkout originally omitted the `model.py` imported by the proposed-model
training scripts. The added `model.py` reimplements **STE + TTE + CSA** from
Section III-B and Fig. 2 of the paper. It is not the recovered author file, and
author/ablation checkpoints are not guaranteed to be compatible.

The prediction path is:

```text
[B, 4, 3, 384, 384]
  -> shared spatial transformer: one 768-dimensional CLS per frame
  -> temporal transformer: temporal CLS plus four frame tokens
  -> temporal cross-attention: aligned CLS query, all five tokens as keys/values
  -> residual-enhanced CLS [B, 768]
  -> Linear + ELU -> [B, 256, 16, 16]
  -> convolution / transpose convolution -> [B, 3, 256, 256]
```

Both encoders default to 12 blocks and 8 heads, with 32x32 patches and a
768-dimensional embedding. The MLP width is 3072, following the existing model
constructor; the paper does not specify that width. Four input frames and the
384px input / 256px target sizes follow the existing data pipeline. Frame order
is preserved, and the spatial encoder shares its parameters across all frames.
The model uses PyTorch directly without downloading pretrained weights or
requiring `timm`. Inputs are normalized to [-1, 1]; the output uses Tanh.

Eq. (11) in the paper defines Q, K and V, but its final expression is ambiguous
about reducing the sequence to a single CLS vector. This implementation uses
standard `softmax(QK^T / sqrt(d_head)) V` aggregation, including the two
alignment projections described in Eq. (11) and the original CLS residual in
Eq. (12). This choice is an explicit interpretation of the paper.

Minimal use:

```python
import torch
from model import VisionTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisionTransformer().to(device).eval()
clip = torch.randn(1, 4, 3, 384, 384, device=device).clamp(-1, 1)
with torch.no_grad():
    predicted_frame = model(clip)  # [1, 3, 256, 256]
```

The existing training configuration still defaults to `b16` (2 layers, 6 heads,
16px patches). Select the new `astt` preset to pass the paper's architecture
parameters through a training script, and disable the old checkpoint for a
fresh model:

```bash
python train_val_UIT_ADrone.py --model-arch astt --image-size 384 --train 1 --checkpoint-path ''
```

That script additionally requires its existing dependencies, dataset directories
and checkpoint output directory to be set up. Its optimizer remains SGD, while
the paper reports AdamW; this command selects the architecture and does not
claim to reproduce the paper's training results. `eval.py` and `check_jax.py`
are legacy classification utilities, not evaluation entry points for this
frame predictor. The prediction training loss is MSE against the fifth frame.

Run CPU structural and gradient checks with:

```bash
python -m unittest discover -s tests -v
```

These checks cover frame order, cross-attention aggregation and the single CLS
residual, gradients through all four stages, output shape/range, invalid inputs,
and the paper-sized default architecture.

## Inference

## Dataset description
> UIT-ADrone dataset [1] consists of 3 different scenes captured by the complex traffic environment in Hochiminh city with 206,194 video frames (59,186 frames for the training set and 147,005 frames for the test set) in total, with size 1920 × 1080. Additionally, the dataset contains 592 training snippets and 905 testing snippets, totaling nearly 6.50 hours. The dataset has a total of 10 types of abnormal events, including crossing the road at the wrong lane, walking under the street, driving in the wrong roundabout, illegally driving on the sidewalk, illegal left turn/ turn right, illegally parking in the street, carrying bulky goods, parking on the sidewalk, driving in the opposite directions, and falling off motorcycles. In addition, the training snippets only have normal events, while testing snippets consist of both normal and unusual events. Additionally, the dataset is challenging for evaluation because of complex light conditions and camera movement. Furthermore, 63,485 ground truth annotations are provided for the testing set in the form of bounding boxes around each anomalous event in each extracted video frame, which helps evaluate the performance.

## References
[1] Tran, Tung Minh and Vu, Tu N and Nguyen, Tam V and Nguyen, Khang, “UIT-ADrone: A Novel Drone Dataset for Traffic Anomaly Detection”, IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, IEEE, vol. 16, pp. 5590–5601, 2023.

## Citation

If this repository proves beneficial for your projects, we kindly request acknowledgment through proper citation:
```
@InProceedings{Tran_2024_TCSVT,
    author    = {Tran, Tung Minh and Bui, Doanh C. and Nguyen, Tam V. and Nguyen, Khang},
    title     = {Transformer-based Spatio-Temporal Unsupervised Traffic Anomaly Detection in Aerial Videos},
    journal = {IEEE Transactions on Circuits and Systems for Video Technology},
    month     = {},
    year      = {},
    pages     = {}
}
```
