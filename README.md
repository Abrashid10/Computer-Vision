[README (1).md](https://github.com/user-attachments/files/33706974/README.1.md)
# Computer Vision

A collection of deep learning and computer vision projects, covering object detection, object tracking, image segmentation, generative models, neural style transfer, and vision transformers.

> **Status:** Work in progress

## Projects

| Folder | Topic | Description |
|--------|-------|-------------|
| [`yolov1/`](./yolov1) | Object detection | YOLOv1 implementation: single-stage, real-time object detection |
| [`yolov3/`](./yolov3) | Object detection | YOLOv3 implementation with multi-scale predictions |
| [`sort_object_tracking/`](./sort_object_tracking) | Object tracking | Multi-object tracking with SORT (Simple Online and Realtime Tracking) |
| [`deepsort_object_tracking/`](./deepsort_object_tracking) | Object tracking | Multi-object tracking with DeepSORT, adding appearance features to SORT |
| [`Image-segmentation/`](./Image-segmentation) | Segmentation | Image segmentation models |
| [`GANS/`](./GANS) | Generative models | Generative Adversarial Networks |
| [`neuralstyle/`](./neuralstyle) | Style transfer | Neural style transfer: apply the style of one image to another |
| [`pytorch_lightning/`](./pytorch_lightning) | Training framework | Computer vision models organized with PyTorch Lightning |
| [`Transformer_image.ipynb`](./Transformer_image.ipynb) | Vision transformers | Notebook applying transformers to images |

## Repository Structure

```
Computer-Vision/
├── GANS/
├── Image-segmentation/
├── deepsort_object_tracking/
├── neuralstyle/
├── pytorch_lightning/
├── sort_object_tracking/
├── yolov1/
├── yolov3/
└── Transformer_image.ipynb
```

## Getting Started

### Prerequisites

- Python 3.8+
- A CUDA-capable GPU is recommended for training (CPU works for small experiments)

### Installation

```bash
git clone https://github.com/Abrashid10/Computer-Vision.git
cd Computer-Vision

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install torch torchvision pytorch-lightning numpy opencv-python matplotlib pillow tqdm jupyter
```

Each project may have additional dependencies. Check the files inside its folder.

### Running a project

Open a project folder and follow the scripts or notebooks inside it. For the notebook:

```bash
jupyter notebook Transformer_image.ipynb
```

## Topics Covered

- **Object detection:** YOLOv1, YOLOv3
- **Object tracking:** SORT, DeepSORT
- **Segmentation:** pixel-level image understanding
- **Generative models:** GANs
- **Style transfer:** content and style image blending
- **Vision transformers:** attention-based image models
- **Tooling:** PyTorch and PyTorch Lightning

## Roadmap

- [ ] Add a README to each project folder with usage instructions
- [ ] Add sample outputs (detections, tracked videos, generated images, stylized images)
- [ ] Add a `requirements.txt`
- [ ] Add pretrained weights or download links
- [ ] Document datasets used by each project

## Author

**Abdallah**  
GitHub: [@Abrashid10](https://github.com/Abrashid10)

## License

Add a license of your choice (for example MIT) and update this section.
