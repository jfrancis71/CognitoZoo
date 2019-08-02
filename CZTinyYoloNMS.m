(* ::Package:: *)

net=Import["~/Google Drive/Personal/Computer Science/CZModels/TinyNMSTraining/2019-07-31T21:56:15_1_023_18400_9.39e-3_9.52e-3.wlnet"];


SetDirectory["~/CognitoZoo"]


<<CZTinyYoloV2Pascal.m


(* ::Input:: *)
(*nmsnet=NetTake[net,{"cond1","log"}];*)


(* ::Input:: *)
(*trunknet=NetTake[net,{"base","base"}];*)


(* ::Input:: *)
(*CZICMStep[ out_, sofar_ ] := ( *)
(*r2=nmsnet[Association["1"->out,"cond"->sofar]];*)
(*m=(1-sofar)*r2;*)
(*If[Max[m]<.24,sofar,CZICMStep[out, ReplacePart[sofar,Position[m,Max[m]]->1]] ]) *)


(* ::Input:: *)
(*CZICM[ out_ ]:= (CZICMStep[ out, ConstantArray[0,{5,13,13}]])*)


(* ::Input:: *)
(*CZOutput[threshold_:.24][image_]:=( *)
(*netOutput=YoloNet[image];*)
(*slots=LogisticSigmoid[netOutput[[5;;105;;25]]]*SoftmaxLayer[][Transpose[Partition[netOutput,25][[All,6;;25]],{1,4,2,3}]];*)
(*nms=Position[CZICM[ trunknet[ image ] ],1];*)
(*Map[{CZGetBoundingBox[#,netOutput],Extract[CZPascalClasses,Ordering[Extract[slots,#],-1]],Max@Extract[slots,#]}&,nms]);*)


(* ::Input:: *)
(*CZDetectObjectsNMS[ image_, opts:OptionsPattern[] ] :=*)
(*   CZObjectsDeconformer[ image, {416, 416}, "Fit" ]@CZOutput[ .24 ]@CZImageConformer[{416,416},"Fit"]@image;*)


(* ::Input:: *)
(*CZHighlightObjectsNMS[img_,opts:OptionsPattern[]]:=HighlightImage[img,CZDisplayObjects@CZDetectObjectsNMS[img,opts]]*)


(* ::Input:: *)
(*CZErrorMatchingRectangles[ rects1_, rects2_ ] := (Length[rects1]+Length[rects2]-2*Total@N@Map[CZIntersectionOverUnion[rects1[[#[[1,1]]]],rects2[[#[[2,1]]]]]&,FindIndependentEdgeSet[ Graph[Flatten@Table[l[i]->r[j],{i,1,Length[rects1]},{j,1,Length[rects2]}], EdgeWeight->Flatten@Table[CZIntersectionOverUnion[rects1[[i]],rects2[[j]]],{i,1,Length[rects1]},{j,1,Length[rects2]}]] ] ])*)


(* ::Input:: *)
(*CZErrorMatchingObjects[ objs1_, objs2_ ] := CZErrorMatchingRectangles[ objs1[[All,1]], objs2[[All,1]] ]*)
