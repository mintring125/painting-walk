# 그림 속 산책 · Walk inside the painting

One painted image per scene becomes a 3D Gaussian splat you can walk around in with a VRM character —
in the browser, no install.

Scenes: seaside town street · shrine stone stairs · seaside train · sunset beach · night alley after rain · living room.

## What you can do

- **Move**: click the ground, or WASD / arrow keys (Shift to run). Stairs are climbed step by step.
- **Interact**: hover shows what an object does; click it to sit (bench, sofa, train seat, step), touch
  (vending machine, mailbox, lantern, books…), look (sea, sunset, window) or pray (shrine).
- **People**: a shopkeeper on the street and a passenger on the train greet you, follow you with their eyes,
  and answer when you click them.
- **Camera**: painting view (exactly the original picture), free view (drag to orbit), split view (both at once).
- Space sits / stands, Q waves, "처음부터" plays a guided stroll, drop a `.vrm` file to change the character.

## Run locally

```bash
python serve.py 8770 .
# open http://localhost:8770  (add ?scene=stairs etc.)
```

Any static file server works; `serve.py` just disables caching.

## How a scene is made

1. Paint or generate an image (eye-level, a flat floor in the bottom third).
2. [Apple SHARP](https://github.com/apple/ml-sharp) turns it into a 3D Gaussian splat (`sharp predict`).
3. `analyze_scene.py` finds the floor, builds an invisible walkable heightfield (obstacles, cracks and
   specks cleaned up, long stairways/roads rebuilt from their centre profile), and places hotspots and NPCs
   from pixel boxes in `scenes/<id>/hotspots.json`. `build_scenes.sh` holds the options for every scene.
4. Splats are compressed to `.sog` with `npx @playcanvas/splat-transform`.
5. `index.html` + `main.js` render with three.js + [Spark](https://sparkjs.dev) + [three-vrm](https://github.com/pixiv/three-vrm).

`stuck_test.py scenes/<id>/scene.json` reports places where keyboard walking would get stuck.

## Credits and licences

- Splats: generated with Apple SHARP. **The SHARP model is licensed for non-commercial research only**
  ([LICENSE_MODEL](https://github.com/apple/ml-sharp/blob/main/LICENSE_MODEL)); this demo is non-commercial.
- Characters: pixiv VRM samples — `VRM1_Constraint_Twist_Sample` (VRM Public License 1.0) and
  `three-vrm-girl` (VRoid Hub licence: redistribution, modification and commercial use allowed, credit not required).
- Background paintings: AI-generated (OpenAI image model) for this project.
- Libraries: three.js (MIT), Spark (MIT), @pixiv/three-vrm (MIT), Nanum Pen Script font (OFL).
- Inspired by monoradio's "絵の中を歩く" demo.
