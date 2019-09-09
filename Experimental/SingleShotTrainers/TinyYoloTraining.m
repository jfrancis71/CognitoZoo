(* ::Package:: *)

(*
   Example code for training a TinyYolo net to do face detection.
   It puts face detection in class 1 (where the aeroplane went in
   original implementation.
   Note there is no box regression training in this example, so
   localisation will be crude.
   This code is site specific, you will need to alter the faces/files import to point
   to a suitable dataset.
   This is an "in-core" trainer which does appropriate object conforming.
*)


<<Experimental/SingleShotTrainers/SingleShotTraining.m
<<CZUtils.m


TinyYoloNet = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/TinyYoloV2.wlnet"],"WLNet"];


boundingBoxNet = NetTake[TinyYoloNet,{"decoderNet","flatBoxes"}];


boundingBoxes = Rectangle@@@boundingBoxNet[ConstantArray[0,{125,13,13}]]["Boxes"];


files1=FileNames["~/ImageDataSets/FaceScrub/ActorImages/VGA/ActorImages1/*.jpg"];
faces1=Import["~/ImageDataSets/FaceScrub/ActorImages/DLibVGAActor1.mx"];


objs = Table[
   CZConformObjects[
      Table[{faces[[i,d]],"Face"},{d,1,Length[faces[[i]]]}],
      Import@files[[i]],{416,416},"Fit"],
   {i,1,1000}];


dataset = Table[
   Append[
      SSDStyleYoloTargets[
         objs[[i]],
         boundingBoxes, 
         Prepend[ ConstantArray["NA",{19}],"Face"]],
      "Input"->CZImageConformer[{416,416},"Fit"]@Import@files[[i]]],
   {i,1,1000}];


{trainingSet,validationSet} = {dataset[[;;900]],dataset[[901;;]]};


lossNet = AttachSingleShotLossLayer[ TinyYoloNet ]


(*
trained=NetTrain[lossNet,trainingSet,ValidationSet\[Rule]validationSet,LossFunction\[Rule]{"Loss2","Loss1"}]
*)
