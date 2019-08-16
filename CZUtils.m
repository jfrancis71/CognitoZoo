(* ::Package:: *)

CZDisplayObjects[ {}, _:{} ] := {}
CZDisplayObjects[ detections_, cmap_:{} ] := MapThread[
  {If[cmap==={},Red,cmap[#2]], #1, Inset[Style[#2,White,FontSize->Scaled[1/50],Background->Black],First[#1],{Left,Bottom}]}&,
   Transpose@detections ];


Options[ CZImageConformer ] = {
   Padding->0.0 };
CZImageConformer[ dims_, fitting_, opts:OptionsPattern[] ][ image_ ] := First@ConformImages[ {image}, dims, fitting , Padding->OptionValue[ Padding ] ];


CZIntersection[a_Rectangle, b_Rectangle] := Module[{xa=Max[a[[1,1]],b[[1,1]]],ya=Max[a[[1,2]],b[[1,2]]],xb=Min[a[[2,1]],b[[2,1]]],yb=Min[a[[2,2]],b[[2,2]]]},
   If[xa>xb||ya>yb,0,(xb-xa)*(yb-ya)]]
CZArea[a_Rectangle] := ( a[[1,1]]-a[[2,1]] ) * ( a[[1,2]]-a[[2,2]] )
CZUnion[a_Rectangle,b_Rectangle] := CZArea[a] + CZArea[b] - CZIntersection[a, b]


(* Had considered using RegionIntersection/RegionUnion but this was overly general and unacceptably slow in practice.
   Not uncommon to see 100 raw detections, hence 10,000 pairs to evaluate.
*)
CZIntersectionOverUnion[a_Rectangle, b_Rectangle]:= 
   CZIntersection[ a, b ] / CZUnion[a, b]


(*
   Note: requires format list of {Rectangle[{xmin,ymin},{xmax,ymax}],(metrics...),prob}
   It is sensitive to that xmin,ymin,xmax,ymax ordering and will not
   work if it is wrong way round (ie corners in wrong order)
   
   This NonMaxSuppression algorithm implements a fairly standard non max suppression algorithm
   with two exceptions. The standard algorithm takes the largest probability detection as the
   result of overlapping detections which is the default here. But you have the option of specifying
   CZTakeWeightedRectangle instead which computes a weighted average (by their probability) of the
   overlapping detections (both metrics and rectangles) so the metrics argument needs to be numeric
   ie can multiply by a scalar and add. In the case of Yolo and SSD it is just an empty list,
   but for example for face detection it can compute gender estimate or anything else chosen to be
   placed there.
*)
CZTakeMaxProbRectangle[ objects_ ] := First@SortBy[objects,-Last[#]&];
CZTakeWeightedRectangle[ objects_ ] :=
{
   Rectangle@@Round[Total[objects[[All,-1]]*List@@@objects[[All,1]]]/Total[objects[[All,-1]]]],
   Max[objects[[All,-1]]],
   If[ Length[objects[[1]]] > 2,
      Round[Total[objects[[All,-1]]*List@@@objects[[All,3]]]/Total[objects[[All,-1]]]],
      Nothing ]
}
CZNonMaxSuppressionMethod[ nonMaxSuppressionMethod_ ][ maxOverlapFraction_ ][ objects_ ] := 
   nonMaxSuppressionMethod /@ Gather[ objects, (CZIntersectionOverUnion[#1[[1]],#2[[1]]]> maxOverlapFraction )& ];
CZNonMaxSuppression = CZNonMaxSuppressionMethod[ CZTakeMaxProbRectangle ];


(*
   Note: requires format list of { Rectangle[{xmin,ymin},{xmax,ymax}],class, prob }
   It is sensitive to that xmin,ymin,xmax,ymax ordering and will not
   work if it is wrong way round (ie corners in wrong order)
*)
(* Does Non Max Suppression seperately by object class *)
SyntaxInformation[ NMSMethod ]= {"ArgumentsPattern"->{_}};
Options[ CZNonMaxSuppressionPerClass ] = {
   NMSMethod->CZNonMaxSuppression,
   MaxOverlapFraction->.25 };
CZNonMaxSuppressionPerClass[opts:OptionsPattern[] ][ objects_ ] :=
   Flatten[
      Map[ Function[ objectsInClass, {#[[1]],objectsInClass[[1,2]],#[[2]]}&/@OptionValue[ NMSMethod ][ OptionValue[ MaxOverlapFraction ]  ][ objectsInClass[[All,{1,3}]] ] ], GatherBy[ objects, #[[2]]& ] ], 1 ]


CZDeconformRectangles[ {}, _, _, _ ] := {};
CZDeconformRectangles[ boxes_List, image_Image, netDims_List, "Fit" ] :=
   Module[{ netAspectRatio = netDims[[2]]/netDims[[1]], padding, scale },
      padding = If [ ImageAspectRatio[image] < netAspectRatio,
         {0,(ImageDimensions[image][[1]]*netAspectRatio-ImageDimensions[image][[2]])/2},
         {(ImageDimensions[image][[2]]*(1/netAspectRatio)-ImageDimensions[image][[1]])/2,0}
         ];
      scale = If [ ImageAspectRatio[image] < netAspectRatio, ImageDimensions[image][[1]]/netDims[[1]], ImageDimensions[image][[2]]/netDims[[2]] ];
      Rectangle@@@Round@Transpose[Transpose[List@@@boxes,{2,3,1}]*scale - padding,{3,1,2}]
];
CZDeconformRectangles[ rboxes_List, image_Image, netDims_List, "Stretch" ] := 
   Module[ {
      boxes = Map[{#[[1]],#[[2]]}&,rboxes] },
      Map[Rectangle[Round[#[[1]]],Round[#[[2]]]]&, Transpose[Transpose[boxes,{2,3,1}]*ImageDimensions[image]/netDims,{3,1,2}]]
];


(* Implicitly assumes that the rectangles are first entry in the list of objects.
   So { {rect1, class1, prob1 }, ... }
*)
CZObjectsDeconformer[ image_Image, netDims_List, fitting_String ][ objects_ ] :=
   Transpose[ { CZDeconformRectangles[ objects[[All,1]], image, netDims, fitting ], objects[[All,2]], objects[[All,3]] } ]


CZConformRectangles[ rboxes_List, image_Image, netDims_List, "Fit" ] :=
   With[{netAspectRatio = netDims[[2]]/netDims[[1]]},
      With[ {
         boxes = Map[{#[[1]],#[[2]]}&,rboxes],
         padding = If [ ImageAspectRatio[image] < netAspectRatio,
            {0,(ImageDimensions[image][[1]]*netAspectRatio-ImageDimensions[image][[2]])/2},
            {(ImageDimensions[image][[2]]*(1/netAspectRatio)-ImageDimensions[image][[1]])/2,0}
            ],
         scale = If [ ImageAspectRatio[image] < netAspectRatio, ImageDimensions[image][[1]]/netDims[[1]], ImageDimensions[image][[2]]/netDims[[2]] ]
         },
         Map[Rectangle[Round[#[[1]]],Round[#[[2]]]]&, Transpose[(Transpose[boxes,{2,3,1}]+ padding)/scale,{3,1,2}]]
   ]];



(*
   Utility. Gives an indication of how many MULADDS are involved in a net.
   Presumably requires the net to know about its input size to calculate this.
*)
CZNetInformation[conv_ConvolutionLayer]:=Apply[Times,NetExtract[conv,"Output"]]*NetExtract[conv,"Input"][[1]]*Apply[Times,NetExtract[conv,"KernelSize"]];
CZNetInformation[elem_ElementwiseLayer]:=0;
CZNetInformation[pool_PoolingLayer]:=0;
CZNetInformation[pad_PaddingLayer]:=0;
CZNetInformation[cat_CatenateLayer]:=0;
CZNetInformation[plus_ConstantPlusLayer]:=0;
CZNetInformation[times_ConstantTimesLayer]:=0;
CZNetInformation[part_PartLayer]:=0;
CZNetInformation[reshape_ReshapeLayer]:=0;
CZNetInformation[soft_SoftmaxLayer]:=0;
CZNetInformation[trans_TransposeLayer]:=0;
CZNetInformation[trans_BatchNormalizationLayer]:=0;
CZNetInformation[net_NetChain]:=Total@Table[CZNetInformation[net[[l]]],{l,1,Length[net]}];
CZNetInformation[net_NetGraph]:=Total@Table[CZNetInformation[net[[l]]],{l,1,Length[net]}]
