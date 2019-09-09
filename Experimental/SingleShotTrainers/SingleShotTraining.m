(* ::Package:: *)

(*
   For each object it returns a list of matched anchors.
   Note different neural nets have made different decisions over the matching strategy.
   So this is just one example.
   This implements the second part of the matching strategy outlined in:
   https://arxiv.org/pdf/1512.02325.pdf See page 5, Matching Strategy
*)
CZObjectsToAnchors[objects_, anchors_]:=
   Map[Function[object,
      Position[Map[CZIntersectionOverUnion[#,object[[1]]]&,anchors],x_/;x>.5][[All,1]]],objects]


(*
   Takes a list of objects of form {{rect1,classname1},{rect2,classname2},....} along
   with a list of all anchorboxes in the same order as output by the net
   and a list of class labels (again in same order as output by the net
   Returns an association with the targets wired up (so you need to add the Input
   association for training).
*)
SSDStyleYoloTargets[ objects_, anchors_, labels_ ] :=
   Module[{matches=CZObjectsToAnchors[objects,anchors]},
      Association[
         "ClassProbs"->ReplacePart[ConstantArray[0,{Length[anchors],Length[labels]}],
            Flatten[MapThread[Function[{matchPerObj,obj},Map[{#,First@First@Position[labels,obj[[2]]]}->1&,matchPerObj]],{matches,objects}],1]],
         "Objectness"->ReplacePart[ConstantArray[0,{Length[anchors]}],(#->1&)/@Flatten[matches]],
         "Boxes"->ConstantArray[0,{Length[anchors],2,2}]]];


AttachSingleShotLossLayer[ baseNet_ ] :=
   NetGraph[{
      "base"->baseNet,"pt1"->PartLayer[{All,1}],"pt2"->PartLayer[{All,1}],
      "mask"->MaskLossLayer,"cr"->CrossEntropyLossLayer["Binary"]},{
      NetPort["base","Objectness"]->NetPort["mask","Mask"],
      NetPort["base","ClassProb"]->"pt1"->NetPort["mask","Input"],
      NetPort["ClassProb"]->"pt2"->NetPort["mask","Target"],
      NetPort["base","Objectness"]->NetPort["cr","Input"],
      NetPort["Objectness"]->NetPort["cr","Target"]
}];


<<Experimental/CZMaskLoss.m
