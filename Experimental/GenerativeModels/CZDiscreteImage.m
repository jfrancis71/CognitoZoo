(* ::Package:: *)

CZDiscretize[image_]:=Map[1+Round[#*9]&,ImageData[image],{2}]


CZOneHot[ image_ ] := Map[ ReplacePart[ ConstantArray[ 0, {10} ], #->1 ]&, image, {2} ];


CZSampleDiscreteImage[ probs_ ] := Map[ RandomChoice[#->Range[1,10]]&, probs, {2} ];


SyntaxInformation[ CZGenerativeModel ]= {"ArgumentsPattern"->{_,_,_,_}};


SyntaxInformation[ CZBinaryVector ]= {"ArgumentsPattern"->{_}};


SyntaxInformation[ CZBinaryImage ]= {"ArgumentsPattern"->{_}};


SyntaxInformation[ CZDiscreteImage ]= {"ArgumentsPattern"->{_}};
