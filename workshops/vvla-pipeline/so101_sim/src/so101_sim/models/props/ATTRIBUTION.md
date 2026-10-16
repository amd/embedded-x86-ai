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
- Copyright: FabHacks contributors
- License: [MIT](https://github.com/merlinyx/fabhacks/blob/main/LICENSE)
- File: `toothbrush.obj`

## Prepared-file checksums

```text
9dc73864bce1992e22d4ea2466540cbe1b049b4a8070d864c26b01e8cb2ce92c  book.obj
d9499e23d4a483b9aa741e0d3aa321ba6baa04be48b1e0c91077685c559aa9cf  book.png
93d89882baaaf966a8113860b9dd7c304793cf9ea7b1f2fc4902490e9559153c  bottle.obj
1e0114f5ce3c98fbf7e14f7bd8c46a995b992d0a92b7370cb5fbd046e0e2c63d  bottle.png
b11e8d25e5a7d1c2bf962921169cda1fa446611efa02140213b0bf00a91ecdec  sports_ball.obj
b146adbcb6098b91f4f5a38de909e8b7998d0aac63d469d18262046e05096d60  sports_ball.png
5ec70d2fdb42f634e079b09843de99c2a9800ee27c3596f5821407a9f5622835  toothbrush.obj
```
