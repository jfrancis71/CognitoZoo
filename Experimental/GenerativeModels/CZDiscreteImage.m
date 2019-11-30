(* ::Package:: *)

CZDiscretize[image_]:=Map[1+Round[#*9]&,ImageData[image],{2}]


CZOneHot[ image_ ] := Transpose[Map[ ReplacePart[ ConstantArray[ 0, {10} ], #->1 ]&, image, {2} ] ,{2,3,1}];


discreteSample1[ v_ ] := ReplacePart[ConstantArray[0,{10}],RandomChoice[ v -> Range[1,10] ]->1]


discreteSample[ image_ ] := Transpose[ Map[ discreteSample1, Transpose[ image, {3,1,2} ], {2}], {2,3,1}]
