# Digital-twin prop assets

The files in this directory are redistributed and modified for the SO-101
MuJoCo digital twin. Meshes were reoriented, recentered, and scaled to
graspable workshop dimensions. Physics uses separate primitive collision
geometries defined in `scene_vvla.xml`.

## Sports ball and bottle

- Sources: YCB `053_mini_soccer_ball` and `006_mustard_bottle`, Google 16k
  textured models
- Project: [YCB Object and Model Set](https://www.ycbbenchmarks.com/)
- Copyright: the YCB dataset contributors
- License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
- Files: `sports_ball.obj`, `sports_ball.png`, `bottle.obj`, `bottle.png`

## Book

- Source: Google Scanned Objects, *Eat to Live: The Amazing Nutrient-Rich
  Program for Fast and Sustained Weight Loss, Revised Edition Book*
- MuJoCo conversion: [kevinzakka/mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects)
- Copyright: Google LLC
- Mesh and texture license: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
- Files: `book.obj`, `book.png`

## Toothbrush

- Source: [`toothbrush.obj`](https://github.com/merlinyx/fabhacks/blob/main/fabhacks/parts/meshes/toothbrush.obj)
- Project: [FabHacks](https://github.com/merlinyx/fabhacks)
- Copyright: Yuxuan Mei
- License: MIT, shipped as [`LICENSE.toothbrush`](LICENSE.toothbrush)
- File: `toothbrush.obj`

## Prepared-file checksums

```text
74c4b1283d0e8f0ae349c3ca55960dd6e5db31623ae8aacbc543e950ade307b5  book.obj
d9499e23d4a483b9aa741e0d3aa321ba6baa04be48b1e0c91077685c559aa9cf  book.png
4f9eadf5522dc64c663e91987f44df997752657138a2bfe73ad0345e31b450cf  bottle.obj
3e003accbfb0a7423809feb0160d6f1326123f65d648cfb55031da5ed49cf67b  bottle.png
a8debe927be5d204d02df82db2cba8a5ef92da4a71a86234f8675e99c43ca464  sports_ball.obj
ff4ee16fc4861117e02fb2ee10aa74c3d2d6482157580e42ea488d975778105a  sports_ball.png
c171de4d6a3d89e92a05b9c89aa915ca3f5384d7a419af855e70c625475b902f  toothbrush.obj
```
