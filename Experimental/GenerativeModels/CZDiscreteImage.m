(* ::Package:: *)

CZDiscretize[image_]:=Map[1+Round[#*9]&,ImageData[image],{2}]


CZOneHot[ image_ ] := Transpose[Map[ ReplacePart[ ConstantArray[ 0, {10} ], #->1 ]&, image, {2} ] ,{2,3,1}];
