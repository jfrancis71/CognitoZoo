(* ::Package:: *)

(* Yolo v3 Open Images *)


leakyReLU = ElementwiseLayer[Ramp[#]+Ramp[-#]*-.1&];


hdfFile = "/Users/julian/yolov3/darknet/Yolov3OpenImages.h5";


YoloConvLayer[ name_, filters_, filterSize_, stride_, side_ ] := Module[{
   weights = Import[hdfFile,{"Datasets",name<>"_weights"}],
   biases = Import[hdfFile,{"Datasets",name<>"_biases"}],
   rm = Import[hdfFile,{"Datasets",name<>"_rolling_mean"}],
   rv = Import[hdfFile,{"Datasets",name<>"_rolling_variance"}],
   scales = Import[hdfFile,{"Datasets",name<>"_scales"}]
},
   NetChain[{
      ConvolutionLayer[filters,{filterSize,filterSize},"Stride"->stride,"Weights"->weights,"Biases"->ConstantArray[0,filters],"PaddingSize"->If[filterSize==1,0,1]],
      BatchNormalizationLayer["Epsilon"->.00001,"Input"->{filters,side,side},"Gamma"->scales,"Beta"->biases,"MovingMean"->rm,"MovingVariance"->rv],
      leakyReLU
}]];


(* Residual block filters and resolution unchanged *)
SmallResidualBlock[ firstLayer_, filters_, size_ ] := NetGraph[{
   YoloConvLayer[ "layer"<>ToString[firstLayer], filters/2, 1, 1, size ],
   YoloConvLayer[ "layer"<>ToString[firstLayer+1], filters, 3, 1, size ],
   ThreadingLayer[ Plus ]},
   { 1->2->3, NetPort["Input"]->3 }];


(* Halves the output resolution and doubles number of filters *)
LargeResidualBlock[ firstLayer_, inputFilters_, inputSize_ ] := NetGraph[{
   YoloConvLayer[ "layer"<>ToString[firstLayer], inputFilters*2, 3, 2, inputSize/2 ],
   YoloConvLayer[ "layer"<>ToString[firstLayer+1], inputFilters, 1, 1, inputSize/2 ],
   YoloConvLayer[ "layer"<>ToString[firstLayer+2], inputFilters*2, 3, 1, inputSize/2 ],
   ThreadingLayer[ Plus ]},
   { 1->2->3->4, 1->4 }];


inp = Import["/Users/julian/yolov3/darknet/Yolov3OpenImages.h5",{"Datasets","/input"}];inp//Dimensions;


multiboxObjDecoder[ anchors_, width_, height_ ] := NetChain[{ReshapeLayer[{anchors,606,width,height}],PartLayer[{All,5}],LogisticSigmoid}]


multiboxClassesDecoder[ anchors_, width_, height_ ] := NetChain[{ReshapeLayer[{anchors,606,width,height}],PartLayer[{All,6;;606}],LogisticSigmoid}]


widthScales = {
   Table[{116,156,373}[[n]],{n,1,3},{19},{19}],
   Table[{30,62,59}[[n]],{n,1,3},{38},{38}],
   Table[{10,16,33}[[n]],{n,1,3},{76},{76}]};


heightScales = {
   Table[{90,198,326}[[n]],{n,1,3},{19},{19}],
   Table[{61,45,119}[[n]],{n,1,3},{38},{38}],
   Table[{13,30,23}[[n]],{n,1,3},{76},{76}]};


multiboxLocationsDecoder[ layerNo_, anchors_, width_, height_ ] := NetGraph[{
   "reshape"->ReshapeLayer[{anchors,606,width,height}],
   "cx"->{ PartLayer[{All,1}],LogisticSigmoid,ConstantPlusLayer[ "Biases"->Table[j,{anchors},{i,0,width-1},{j,0,height-1}] ], ElementwiseLayer[608*#/width&] },
   "cy"->{ PartLayer[{All,2}],LogisticSigmoid,ConstantPlusLayer[ "Biases"->Table[i,{anchors},{i,0,width-1},{j,0,height-1}] ], ElementwiseLayer[608*(1-#/height)&] },
   "width"->{ PartLayer[{All,3}], ElementwiseLayer[ Exp ], ConstantTimesLayer[ "Scaling"->widthScales[[layerNo]] ] },
   "height"->{ PartLayer[{All,4}], ElementwiseLayer[ Exp ], ConstantTimesLayer[ "Scaling"->heightScales[[layerNo]] ] },
   "minx"->ThreadingLayer[#1-#2/2&],
   "miny"->ThreadingLayer[#1-#2/2&],
   "maxx"->ThreadingLayer[#1+#2/2&],
   "maxy"->ThreadingLayer[#1+#2/2&],
   "cat"->CatenateLayer[],"reshape1"->ReshapeLayer[{4,anchors,width,height}]
   },{
   "reshape"->{"cx","cy","width","height"},
   {"cx","width"}->"minx",{"cx","width"}->"maxx",{"cy","height"}->"miny",{"cy","height"}->"maxy",
   {"minx","miny","maxx","maxy"}->"cat"->"reshape1"
}]


yoloConvNet = NetGraph[{
   "layer0"->YoloConvLayer[ "layer0", 32, 3, 1, 608],
   "layer4"->LargeResidualBlock[ 1, 32, 608 ],
   "layer8"->LargeResidualBlock[ 5, 64, 304 ],
   "layer11"->SmallResidualBlock[ 9, 128, 152 ],
   "layer15"->LargeResidualBlock[ 12, 128, 152 ],
   "layer18"->SmallResidualBlock[ 16, 256, 76 ],
   "layer21"->SmallResidualBlock[ 19, 256, 76 ],
   "layer24"->SmallResidualBlock[ 22, 256, 76 ],
   "layer27"->SmallResidualBlock[ 25, 256, 76 ],
   "layer30"->SmallResidualBlock[ 28, 256, 76 ],
   "layer33"->SmallResidualBlock[ 31, 256, 76 ],
   "layer36"->SmallResidualBlock[ 34, 256, 76 ],
   "layer40"->LargeResidualBlock[ 37, 256, 76  ],
   "layer43"->SmallResidualBlock[ 41, 512, 38 ],
   "layer46"->SmallResidualBlock[ 44, 512, 38 ],
   "layer49"->SmallResidualBlock[ 47, 512, 38 ],
   "layer52"->SmallResidualBlock[ 50, 512, 38 ],
   "layer55"->SmallResidualBlock[ 53, 512, 38 ],
   "layer58"->SmallResidualBlock[ 56, 512, 38 ],
   "layer61"->SmallResidualBlock[ 59, 512, 38 ],
   "layer65"->LargeResidualBlock[ 62, 512, 38 ],
   "layer68"->SmallResidualBlock[ 66, 1024, 19 ],
   "layer71"->SmallResidualBlock[ 69, 1024, 19 ],
   "layer74"->SmallResidualBlock[ 72, 1024, 19 ],
   "layer75"->YoloConvLayer[ "layer75", 512, 1, 1, 19 ],
   "layer76"->YoloConvLayer[ "layer76", 1024, 3, 1, 19 ],
   "layer77"->YoloConvLayer[ "layer77", 512, 1, 1, 19 ],
   "layer78"->YoloConvLayer[ "layer78", 1024, 3, 1, 19 ],
   "layer79"->YoloConvLayer[ "layer79", 512, 1, 1, 19 ],
   "layer80"->YoloConvLayer[ "layer80", 1024, 3, 1, 19 ],
   "layer81"->ConvolutionLayer[ 1818, {1,1}, "Weights"->Import[hdfFile,{"Datasets","layer81_weights"}], "Biases"->Import[hdfFile,{"Datasets","layer81_biases"}]],
   "layer84"->YoloConvLayer[ "layer84", 256, 1, 1, 19],
   "layer85"->DeconvolutionLayer[ 256, {2,2}, "Weights"->Table[If[j==i,1,0],{j,1,256},{i,1,256},{2},{2}], "Biases"->ConstantArray[0,256],"Stride"->2],
   "layer86"->CatenateLayer[],
   "layer87"->YoloConvLayer["layer87", 256, 1, 1, 38 ],
   "layer88"->YoloConvLayer["layer88", 512, 3, 1, 38 ],
   "layer89"->YoloConvLayer["layer89", 256, 1, 1, 38 ],
   "layer90"->YoloConvLayer["layer90", 512, 3, 1, 38 ],
   "layer91"->YoloConvLayer["layer91", 256, 1, 1, 38 ],
   "layer92"->YoloConvLayer["layer92", 512, 3, 1, 38 ],
   "layer93"->ConvolutionLayer[ 1818, {1,1}, "Weights"->Import[hdfFile,{"Datasets","layer93_weights"}], "Biases"->Import[hdfFile,{"Datasets","layer93_biases"}]],
   "layer96"->YoloConvLayer[ "layer96", 128, 1, 1, 38],
   "layer97"->DeconvolutionLayer[ 128, {2,2}, "Weights"->Table[If[j==i,1,0],{j,1,128},{i,1,128},{2},{2}], "Biases"->ConstantArray[0,128],"Stride"->2],
   "layer98"->CatenateLayer[],
   "layer99"->YoloConvLayer[ "layer99", 128, 1, 1, 76 ],
   "layer100"->YoloConvLayer[ "layer100",256, 3, 1, 76 ],
   "layer101"->YoloConvLayer[ "layer101",128, 1, 1, 76 ],
   "layer102"->YoloConvLayer[ "layer102",256, 3, 1, 76 ],
   "layer103"->YoloConvLayer[ "layer103",128, 1, 1, 76 ],
   "layer104"->YoloConvLayer[ "layer104",256, 3, 1, 76 ],
   "layer105"->ConvolutionLayer[ 1818, {1,1}, "Weights"->Import[hdfFile,{"Datasets","layer105_weights"}], "Biases"->Import[hdfFile,{"Datasets","layer105_biases"}]]
},{
   "layer0"->"layer4"->"layer8"->"layer11"->"layer15"->"layer18"->"layer21"->"layer24"->"layer27"->
   "layer30"->"layer33"->"layer36"->"layer40"->"layer43"->"layer46"->"layer49"->"layer52"->"layer55"->"layer58"->
   "layer61"->"layer65"->"layer68"->"layer71"->"layer74"->"layer75"->"layer76"->"layer77"->"layer78"->"layer79"->"layer80"->"layer81",
   "layer79"->"layer84"->"layer85",
   {"layer85","layer61"}->"layer86"->"layer87"->"layer88"->"layer89"->"layer90"->"layer91"->"layer92"->"layer93",
   "layer91"->"layer96"->"layer97",
   {"layer97","layer36"}->"layer98"->"layer99"->"layer100"->"layer101"->"layer102"->"layer103"->"layer104"->"layer105",
   "layer81"->NetPort["Layer81"],"layer93"->NetPort["Layer93"],"layer105"->NetPort["Layer105"]
}];


yoloDecoderNet = NetGraph[{
   "objmap1"->multiboxObjDecoder[ 3, 19, 19 ],
   "classesmap1"->multiboxClassesDecoder[ 3, 19, 19 ],
   "locationsmap1"->multiboxLocationsDecoder[ 1, 3, 19, 19 ],
   "objmap2"->multiboxObjDecoder[ 3, 38, 38 ],
   "classesmap2"->multiboxClassesDecoder[ 3, 38, 38 ],
   "locationsmap2"->multiboxLocationsDecoder[ 2, 3, 38, 38 ],
   "objmap3"->multiboxObjDecoder[ 3, 76, 76 ],
   "classesmap3"->multiboxClassesDecoder[ 3, 76, 76 ],
   "locationsmap3"->multiboxLocationsDecoder[ 3, 3, 76, 76 ]
   },{
   NetPort["Layer81"]->{"objmap1","classesmap1","locationsmap1"},
   NetPort["Layer93"]->{"objmap2","classesmap2","locationsmap2"},
   NetPort["Layer105"]->{"objmap3","classesmap3","locationsmap3"},
   "objmap1"->NetPort["ObjMap1"], "classesmap1"->NetPort["ClassesMap1"],"locationsmap1"->NetPort["Locations1"],
   "objmap2"->NetPort["ObjMap2"], "classesmap2"->NetPort["ClassesMap2"],"locationsmap2"->NetPort["Locations2"],
   "objmap3"->NetPort["ObjMap3"], "classesmap3"->NetPort["ClassesMap3"],"locationsmap3"->NetPort["Locations3"]
}];


yoloConcatNet = NetGraph[{
   "lt1"->{TransposeLayer[{3->1,4->2,3->4}],FlattenLayer[2]},
   "lt2"->{TransposeLayer[{3->1,4->2,3->4}],FlattenLayer[2]},
   "lt3"->{TransposeLayer[{3->1,4->2,3->4}],FlattenLayer[2]},
   "lcat"->CatenateLayer[],
   "boxes"->ReshapeLayer[{22743,2,2}],
   "ot1"->{TransposeLayer[{3->1,1->2}],FlattenLayer[]},
   "ot2"->{TransposeLayer[{3->1,1->2}],FlattenLayer[]},
   "ot3"->{TransposeLayer[{3->1,1->2}],FlattenLayer[]},
   "ocat"->CatenateLayer[],
   "ct1"->{TransposeLayer[{1->3,2->4}],FlattenLayer[2]},
   "ct2"->{TransposeLayer[{1->3,2->4}],FlattenLayer[2]},
   "ct3"->{TransposeLayer[{1->3,2->4}],FlattenLayer[2]},
   "ccat"->CatenateLayer[]
   },{
   NetPort["Locations1"]->"lt1",
   NetPort["Locations2"]->"lt2",
   NetPort["Locations3"]->"lt3",   
   {"lt1","lt2","lt3"}->"lcat"->"boxes"->NetPort["Locations"],
   NetPort["ObjMap1"]->"ot1",
   NetPort["ObjMap2"]->"ot2",
   NetPort["ObjMap3"]->"ot3",   
   {"ot1","ot2","ot3"}->"ocat"->NetPort["ObjMap"],
   NetPort["ClassesMap1"]->"ct1",
   NetPort["ClassesMap2"]->"ct2",
   NetPort["ClassesMap3"]->"ct3",   
   {"ct1","ct2","ct3"}->"ccat"->NetPort["Classes"]
}];


classes=Import["/Users/julian/yolov3/darknet/data/openimages.names","List"];


yoloOpenImagesNet = NetGraph[{
   yoloConvNet,
   yoloDecoderNet,
   yoloConcatNet},{
   NetPort[1,"Layer81"]->NetPort[2,"Layer81"],
   NetPort[1,"Layer93"]->NetPort[2,"Layer93"],
   NetPort[1,"Layer105"]->NetPort[2,"Layer105"],
   NetPort[2,"ObjMap1"]->NetPort[3,"ObjMap1"],
   NetPort[2,"ObjMap2"]->NetPort[3,"ObjMap2"],
   NetPort[2,"ObjMap3"]->NetPort[3,"ObjMap3"],
   NetPort[2,"ClassesMap1"]->NetPort[3,"ClassesMap1"],
   NetPort[2,"ClassesMap2"]->NetPort[3,"ClassesMap2"],
   NetPort[2,"ClassesMap3"]->NetPort[3,"ClassesMap3"],
   NetPort[2,"Locations1"]->NetPort[3,"Locations1"],
   NetPort[2,"Locations2"]->NetPort[3,"Locations2"],
   NetPort[2,"Locations3"]->NetPort[3,"Locations3"]},
   "Input"->NetEncoder[{"Image",{608,608},"ColorSpace"->"RGB"}]];


<<CZUtils.m


CZOutputDecoder[ threshold_:.5 ][ output_ ] := Module[{
   detectionBoxes = Union@Flatten@Position[(tmp2=output["ObjMap"])*(tmp1=output["Classes"]),x_/;x>threshold][[All,1]]},det1=detectionBoxes;
   Map[ {
      Rectangle@@output["Locations"][[#]],
      Transpose[{ 
         classes[[Flatten@Position[output["Classes"][[#]]*output["ObjMap"][[#]],x_/;x>threshold] ]],
         Extract[output["Classes"][[#]], Position[output["Classes"][[#]]*output["ObjMap"][[#]],x_/;x>threshold] ]
       }] }&, detectionBoxes ]
];


CZNonMaxSuppression[ nmsThreshold_ ][ dets_ ] := Module[ { deletions },
   deletions = Table[
      Max[
         Table[If[d!=d1&&dets[[d,2,r,1]]==dets[[d1,2,r1,1]]&&CZIntersectionOverUnion[dets[[d,1]],dets[[d1,1]]]>nmsThreshold&&dets[[d,2,r,2]]<dets[[d1,2,r1,2]],1,0],
            {d1,1,Length[dets]},{r1,1,Length[dets[[d1,2]]]}]]
      ,{d,1,Length[dets]},{r,1,Length[dets[[d,2]]]}];
   DeleteCases[Delete[dets, Map[{#[[1]],2,#[[2]]}&,Position[deletions,1]]], {_,{}}]
];


CZDetectionsDeconformer[ image_Image, netDims_List, fitting_String ][ objects_ ] :=
   Transpose[ { CZDeconformRectangles[ objects[[All,1]], image, netDims, fitting ], objects[[All,2]] } ];


Options[ CZDetectObjects ] = {
   TargetDevice->"CPU",
   Threshold->.5,
   NMSIntersectionOverUnionThreshold->.45
};
CZDetectObjects[ image_Image , opts:OptionsPattern[] ] := (
   CZNonMaxSuppression[ OptionValue[ NMSIntersectionOverUnionThreshold ] ]@CZDetectionsDeconformer[ image, {608, 608}, "Fit" ]@CZOutputDecoder[ OptionValue[ Threshold ] ]@
   (yoloOpenImagesNet[ #, TargetDevice->OptionValue[ TargetDevice ] ]&)@CZImageConformer[{608,608},"Fit"]@image
)


Options[ CZHighlightObjects ] = Options[ CZDetectObjects ];
CZHighlightObjects[ image_Image, opts:OptionsPattern[]  ] :=
   HighlightImage[ image, CZDisplayObject/@CZDetectObjects[ image, opts ] ];
