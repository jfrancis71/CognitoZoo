(* ::Package:: *)

(* ::Input:: *)
(*pt=NetGraph[<|"negt"->ElementwiseLayer[1-#&],"negi"->ElementwiseLayer[1-#&],*)
(*"falsebr"->ThreadingLayer[Times],"truebr"->ThreadingLayer[Times],"sum"->ThreadingLayer[Plus]|>,*)
(*{NetPort["Target"]->"negt",NetPort["Input"]->"negi",*)
(*{"negt","negi"}->"falsebr",*)
(*{NetPort["Target"],NetPort["Input"]}->"truebr",*)
(*{"falsebr","truebr"}->"sum"*)
(*}*)
(*];*)


(* ::Input:: *)
(*FocusLossLayer=NetGraph[<|"pt"->pt,"sq"->ElementwiseLayer[-((1-#)^2)&],"crossentropy"->ElementwiseLayer[Log],"times"->ThreadingLayer[Times]|>,*)
(*{"pt"->"sq",*)
(*"pt"->"crossentropy",*)
(*{"sq","crossentropy"}->"times",*)
(*"times"->NetPort["Loss"]*)
(*}];*)


Alpha=NetGraph[{"focus"->FocusLossLayer,"pos"->ElementwiseLayer[#*20&],"neg"->ElementwiseLayer[(1-#)&],"weighting"->ThreadingLayer[Plus],"alpha"->ThreadingLayer[Times]},
{NetPort["Target"]->{NetPort[{"focus","Target"}],"pos","neg"},{"pos","neg"}->"weighting",{NetPort["focus","Loss"],"weighting"}->"alpha"->NetPort["Loss"]}];


(* Computes a masked logistic loss layer, you supply Input and Target and Mask and it computes logistic loss but only for entries where Mask = 1 *)
(* The total loss is the average loss over the number of inputs (regardless of mask *)
MaskLossLayer = NetGraph[ <| 
   "t1"->ThreadingLayer[Times], 
   "t2"->ThreadingLayer[Times], 
   "meancrossentropy"->CrossEntropyLossLayer["Binary"]|>,{
   {NetPort["Input"],NetPort["Mask"]}->"t1", 
   {NetPort["Target"],NetPort["Mask"]}->"t2", 
   "t1"->NetPort[{"meancrossentropy","Input"}], 
   "t2"->NetPort[{"meancrossentropy","Target"}]}
];
