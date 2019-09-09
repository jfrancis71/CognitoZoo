(* ::Package:: *)

<<Experimental/SingleShotTrainers/SingleShotTraining.m


TinyYoloNet = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/TinyYoloV2.wlnet"],"WLNet"];


boundingBoxNet = NetTake[TinyYoloNet,{"decoderNet","flatBoxes"}];


boundingBoxes = Rectangle@@@boundingBoxNet[ConstantArray[0,{125,13,13}]]["Boxes"];


objs = Table[CZConformObjects[Table[{faces[[i,d]],"Face"},{d,1,Length[faces[[i]]]}],Import@files[[i]],{416,416},"Fit"],{i,1,1000}];


dataset = Table[Append[SSDStyleYoloTargets[ objs[[i]], boundingBoxes, Prepend[ ConstantArray["NA",{19}],"Face"]],"Input"->CZImageConformer[{416,416},"Fit"]@Import@files[[i]]],{i,1,1000}];


{trainingSet,validationSet} = {dataset[[;;900]],dataset[[901;;]]};


lossNet = AttachSingleShotLossLayer[ TinyYoloNet ]


(*
trained=NetTrain[lossNet,trainingSet,ValidationSet\[Rule]validationSet,LossFunction\[Rule]{"Loss2","Loss1"}]
*)
