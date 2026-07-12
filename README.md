# cdcEditor
Blender modding tools for Tomb Raider Legend, Anniversary and Underworld.

Allows to import character models, levels, animations, textures and cutscenes, with their animation and audio data.

✓ = Import available

✓✓ = Import and export available

| Tomb Raider Legend/Anniversary             | PC | PS2 | PS3 | PSP | Xbox | Xbox 360 | Gamecube | Wii |
| ------------------------------------------ | -- | ----| ----| ----| -----| ---------| ---------| ----|
| Model                                      | ✓✓  | ✓✓  | ✓   | ✓   | ✓   | ✓        | ✓        | ✓  |
| Next Gen Model                             | ✓✓  |  N/A  |  N/A   | N/A    | N/A    |   N/A       |   N/A       |  N/A  |
| Level                                      | ✓✓  |    |      | ✓   |     |          |          |    |
| Animation                                  | ✓✓  | ✓✓  | ✓✓   | ✓✓   | ✓✓   | ✓✓        | ✓✓        | ✓✓  |
| Cutscene                                   | ✓✓  | ✓  | ✓   | ✓   | ✓   | ✓        | ✓        | ✓  |
| Audio                                      | ✓✓  | ✓  | ✓   | ✓   | ✓   | ✓        | ✓        | ✓  |

| Tomb Raider Underworld                     | PC | PS2 | PS3 | Xbox 360 | Wii |
| ------------------------------------------ | -- | ----| ----| ---------| -----|
| Model                                      | ✓~  | ✓  |     |          |     |
| Level                                      |    |    |      |          |     |
| Animation                                  | ✓✓  | ✓  | ✓   | ✓         | ✓   |
| Cutscene                                   | ✓  | ✓  | ✓   | ✓         | ✓   |
| Audio                                      |    |    |     |           |     |

---
# Model Editing
<img width="1920" height="1080" alt="Desktop Screenshot 2026 07 06 - 20 58 34 94" src="https://github.com/user-attachments/assets/a2cf86f0-3955-4fa6-a74e-8fed6213537b" />
You can import models and edit geometry, materials, skinning and bones. This includes the ability to replace model geometry and materials to the user's liking.

Bones can be added and animated in gameplay or cutscene animations, or used for cloth physics.

---
# Level Editing
https://github.com/user-attachments/assets/635d1063-a502-47e9-9694-a9f10cd3707d

You can import levels and edit geometry, materials, collision, signals, object placement, lights, post processing and more.

---
# Cutscene Editing
https://github.com/user-attachments/assets/25e017f7-329b-407e-a870-dddc8105b84a

You can import cutscenes, and import the animations relative to the models.

Currently, the only way to recognize which InstanceID refers to which character except Lara (whom is always -1), is to run the game and the desired cutscene, and look at the InstanceIDs through Indra's TRLAU Hook.

---
# Known Issues

- Not every level exports correctly and the game may crash

---
# Credits
- [TheIndra](https://github.com/theindra55) for offering support, years of research on the file formats and player ScrollInfo fix
- [arc](https://github.com/arcusmaximus) for offering valuable information
- Che for animation format research
- Joschka for offering valuable information and code from the Noesis plugin for model weight data
