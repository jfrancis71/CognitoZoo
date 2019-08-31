(* ::Package:: *)

TinyYoloNet = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/TinyYoloV2.wlnet"],"WLNet"];


boundingBoxNet=NetTake[TinyYoloNet,{"decoderNet","flatBoxes"}];


boundingBoxes=Rectangle@@@boundingBoxNet[ConstantArray[0,{125,13,13}]]["Boxes"];


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


SSDStyleYoloTargets[ objects_, anchors_, labels_ ] :=
   Module[{matches=CZObjectsToAnchors[objects,anchors]},
      Association[
         "ClassProbs"->ReplacePart[ConstantArray[0,{845,20}],
            Flatten[MapThread[Function[{matchPerObj,obj},Map[{#,First@First@Position[labels,obj[[2]]]}->1&,matchPerObj]],{matches,objects}],1]],
         "Objectness"->ReplacePart[ConstantArray[0,{845}],(#->1&)/@Flatten[matches]],
         "Boxes"->ConstantArray[0,{845,2,2}]]];
