- There clearly seems to be problem with detection and bottle orientations

- Orientations augmentation for 90, 180 and 270 degrees works reliably

- Orientations augmentation for other angles doesn't work

- Solution could be, save several orientations without augmentation and manually label them using LabelImg

- It seems there is a need for third label, "Conveyor". It is needed because in almost all cases fallen covers part of conveyor and NN is learning conveyor belt as fallen bottle


### Ideas:

- Long bottles fallen at some some are very difficult to annotate because to of length. Solution could be to always annotation bottom (towards bottom) part of fallen bottle.