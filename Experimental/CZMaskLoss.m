(* ::Package:: *)

(* Computes a masked logistic loss layer, you supply Input and Target and Mask and it computes logistic loss but only for entries where Mask = 1 *)
(* The total loss is the average loss over the number of inputs (regardless of number of mask entries *)
MaskLossLayer = NetGraph[ <| 
   "t1"->ThreadingLayer[Times], 
   "t2"->ThreadingLayer[Times], 
   "meancrossentropy"->CrossEntropyLossLayer["Binary"]|>,{
   {NetPort["Input"],NetPort["Mask"]}->"t1", 
   {NetPort["Target"],NetPort["Mask"]}->"t2", 
   "t1"->NetPort[{"meancrossentropy","Input"}], 
   "t2"->NetPort[{"meancrossentropy","Target"}]}
];
