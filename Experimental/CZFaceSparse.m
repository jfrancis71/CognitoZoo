(* ::Package:: *)

(*
   Highly experimental, attempt at using sparse distributed cods to describe image
*)


<<CZUtils.m


CZSparseNet=Import["~/Google Drive/Personal/Computer Science/CZModels/SparseFace/2019-08-20T09:00:05_1_08_51038_4.00e-2_5.41e-2.wlnet"];


codebook=Import["~/Google Drive/Personal/Computer Science/CZModels/SparseFace/codebook1.mx"];


CZMakeArrays[ netOutput_ ] := {
   Flatten[ Table[UnitStep[1-Table[CosineDistance[netOutput[[All,cy,cx]],codebook[[1,y,x]]],{y,1,8},{x,1,8}]-0.95],{cy,1,2},{cx,1,2}], { {1,3}, {2,4} } ],
   Flatten[ Table[UnitStep[1-Table[CosineDistance[netOutput[[All,cy,cx]],codebook[[2,y,x]]],{y,1,4},{x,1,4}]-0.95],{cy,1,2},{cx,1,2}], { {1,3}, {2,4} } ],
   Flatten[ Table[UnitStep[1-Table[CosineDistance[netOutput[[All,cy,cx]],codebook[[3,y,x]]],{y,1,4},{x,1,4}]-0.95],{cy,1,2},{cx,1,2}], { {1,3}, {2,4} } ]
};


CZMakeArrays[ netOutput_ ] := {
   Flatten[ Table[UnitStep[Table[Apply[Times,Extract[netOutput[[All,cy,cx]],Position[codebook[[1,y,x]],1]]],{y,1,8},{x,1,8}]-.05],{cy,1,2},{cx,1,2}], { {1,3}, {2,4} } ],
   Flatten[ Table[UnitStep[Table[Apply[Times,Extract[netOutput[[All,cy,cx]],Position[codebook[[2,y,x]],1]]],{y,1,4},{x,1,4}]-.05],{cy,1,2},{cx,1,2}], { {1,3}, {2,4} } ],
   Flatten[ Table[UnitStep[Table[Apply[Times,Extract[netOutput[[All,cy,cx]],Position[codebook[[3,y,x]],1]]],{y,1,4},{x,1,4}]-.05],{cy,1,2},{cx,1,2}], { {1,3}, {2,4} } ]
};


CZDecodeArrays[ result_ ] := Join[
   Map[
         Rectangle[{32*(#[[2]]-.5),512-32*(#[[1]]-.5)}-{37,37},{32*(#[[2]]-.5),512-32*(#[[1]]-.5)}+{37,37}]&,
      Position[result[[1]],x_/;x>0]],
   Map[   
      Rectangle[{64*(#[[2]]-.5),512-64*(#[[1]]-.5)}-{65,65},{64*(#[[2]]-.5),512-64*(#[[1]]-.5)}+{65,65}]&,
      Position[result[[2]],x_/;x>0]],
   Map[
      Rectangle[{64*(#[[2]]-.5),512-64*(#[[1]]-.5)}-{100,100},{64*(#[[2]]-.5),512-64*(#[[1]]-.5)}+{100,100}]&,
      Position[result[[3]],x_/;x>0]]
];      


CZDetectFaces[ image_ ] := CZDeconformRectangles[ CZDecodeArrays@CZMakeArrays@CZSparseNet[ CZImageConformer[{512,512},"Fit"]@image ], image, {512, 512}, "Fit" ];


CZHighlightFaces[ image_ ] := HighlightImage[ image, CZDetectFaces[ image ] ]
