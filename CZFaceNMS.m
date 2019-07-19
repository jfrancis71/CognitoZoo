(* ::Package:: *)

(*
   Highly experimental, attempt at dispensing with NMS
*)


<<CZUtils.m


CZICM[ img_ ] := CZICMStep[ img, {ConstantArray[0,{15,20}],ConstantArray[0,{8,10}],ConstantArray[0,{8,10}]} ];


CZNMSNet = Import["~/Google Drive/Personal/Computer Science/CZModels/NMSNetTraining/2019-07-18T15:50:20_3_01_07084_1.36e-2_4.78e-3.wlnet"];


CZICMStep[ img_, sofar_ ] := (
   s1=sofar[[1]];m1=sofar[[2]];l1=sofar[[3]];
   r=CZNMSNet[Association[
      "Input"->img,
      "GTFaceArray1"->s1,
      "GTFaceArray2"->m1,
      "GTFaceArray3"->l1]];
   m={(1-s1)*r["FaceArray1"],(1-m1)*r["FaceArray2"],(1-l1)*r["FaceArray3"]};
   If[Max[m]<.5,
      {s1,m1,l1},
      l=Position[m,Max[m]]; CZICMStep[img,ReplacePart[sofar,l->1]]]
);


CZDecodeArrays[ result_ ] := Join[
   Map[
         Rectangle[{32*(#[[2]]-.5),480-32*(#[[1]]-.5)}-{37,37},{32*(#[[2]]-.5),480-32*(#[[1]]-.5)}+{37,37}]&,
      Position[result[[1]],x_/;x>0]],
   Map[   
      Rectangle[{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}-{65,65},{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}+{65,65}]&,
      Position[result[[2]],x_/;x>0]],
   Map[
      Rectangle[{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}-{100,100},{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}+{100,100}]&,
      Position[result[[3]],x_/;x>0]]
];      


CZDetectFaces[ image_ ] := CZDeconformRectangles[ CZDecodeArrays@CZICM[ CZImageConformer[{640,480},"Fit"]@image ], image, {640, 480}, "Fit" ];


CZHighlightFaces[ image_ ] := HighlightImage[ image, CZDetectFaces[ image ] ]
