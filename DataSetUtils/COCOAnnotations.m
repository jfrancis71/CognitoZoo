(* ::Package:: *)

cocoAnnotations=Import["~/ImageDataSets/COCO/annotations/instances_val2017.json"];


cocoLabels=Import["~/ImageDataSets/COCO/annotations/coco-paper-labels.txt","List"];


retrieve[imageid_]:=Module[{height=cocoAnnotations[[3,2,Position[cocoAnnotations[[3,2,All,8,2]],imageid][[1,1]],4,2]]},k=height;
({Rectangle[{#[[5,2,1]],height-#[[5,2,2]]},{#[[5,2,1]]+#[[5,2,3]],height-(#[[5,2,2]]+#[[5,2,4]])}],cocoLabels[[#[[6,2]]]]}&)/@(cocoAnnotations[[4,2,#]]&)/@(Flatten@Position[cocoAnnotations[[4,2,All,4,2]],imageid])
]
