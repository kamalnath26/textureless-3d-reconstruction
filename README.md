## Sample commands to use

From a folder of images:

```bash
python reconstruction.py --mode folder --input ./my_images/ --output scene.ply
```

From webcam (real-time SLAM-like):

```bash
python reconstruction.py --mode camera --camera 0 --output scene.ply
```

Using USB camera:

```bash
python reconstruction.py --mode camera --camera 1 --output scene.ply
```