(* ::Package:: *)

CZDiscretize[image_]:=Map[1+Round[#*9]&,ImageData[image],{2}]


CZOneHot[ image_ ] := Map[ ReplacePart[ ConstantArray[ 0, {10} ], #->1 ]&, image, {2} ];


CZSampleDiscreteImage[ probs_ ] := Map[ RandomChoice[#->Range[1,10]]&, probs, {2} ];
