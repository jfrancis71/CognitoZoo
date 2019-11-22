(* ::Package:: *)

SyntaxInformation[ DiscreteImage ]= {"ArgumentsPattern"->{_,_}};


ToDiscreteImage[image_,levels_]:=DiscreteImage[levels, Round[ ImageData[image]*levels ] ]
