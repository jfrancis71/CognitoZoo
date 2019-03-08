(* ::Package:: *)

(* Yolo v3 Open Images *)


leakyReLU = ElementwiseLayer[Ramp[#]+Ramp[-#]*-.1&];


YoloConvLayer[ filters_, filterSize_, stride_, side_ ] :=
   NetChain[{
      ConvolutionLayer[filters,{filterSize,filterSize},"Stride"->stride,"Biases"->ConstantArray[0,filters],"PaddingSize"->If[filterSize==1,0,1]],
      BatchNormalizationLayer["Epsilon"->.00001,"Input"->{filters,side,side}],
      leakyReLU
}];


(* Residual block filters and resolution unchanged *)
SmallResidualBlock[ filters_, size_ ] := NetGraph[{
   YoloConvLayer[ filters/2, 1, 1, size ],
   YoloConvLayer[ filters, 3, 1, size ],
   ThreadingLayer[ Plus ]},
   { 1->2->3, NetPort["Input"]->3 }];


(* Halves the output resolution and doubles number of filters *)
LargeResidualBlock[ inputFilters_, inputSize_ ] := NetGraph[{
   YoloConvLayer[ inputFilters*2, 3, 2, inputSize/2 ],
   YoloConvLayer[ inputFilters, 1, 1, inputSize/2 ],
   YoloConvLayer[ inputFilters*2, 3, 1, inputSize/2 ],
   ThreadingLayer[ Plus ]},
   { 1->2->3->4, 1->4 }];


yoloConvNet = NetGraph[{
   "layer0"->YoloConvLayer[ 32, 3, 1, 608],
   "layer4"->LargeResidualBlock[ 32, 608 ],
   "layer8"->LargeResidualBlock[ 64, 304 ],
   "layer11"->SmallResidualBlock[ 128, 152 ],
   "layer15"->LargeResidualBlock[ 128, 152 ],
   "layer18"->SmallResidualBlock[ 256, 76 ],
   "layer21"->SmallResidualBlock[ 256, 76 ],
   "layer24"->SmallResidualBlock[ 256, 76 ],
   "layer27"->SmallResidualBlock[ 256, 76 ],
   "layer30"->SmallResidualBlock[ 256, 76 ],
   "layer33"->SmallResidualBlock[ 256, 76 ],
   "layer36"->SmallResidualBlock[ 256, 76 ],
   "layer40"->LargeResidualBlock[ 256, 76  ],
   "layer43"->SmallResidualBlock[ 512, 38 ],
   "layer46"->SmallResidualBlock[ 512, 38 ],
   "layer49"->SmallResidualBlock[ 512, 38 ],
   "layer52"->SmallResidualBlock[ 512, 38 ],
   "layer55"->SmallResidualBlock[ 512, 38 ],
   "layer58"->SmallResidualBlock[ 512, 38 ],
   "layer61"->SmallResidualBlock[ 512, 38 ],
   "layer65"->LargeResidualBlock[ 512, 38 ],
   "layer68"->SmallResidualBlock[ 1024, 19 ],
   "layer71"->SmallResidualBlock[ 1024, 19 ],
   "layer74"->SmallResidualBlock[ 1024, 19 ],
   "layer75"->YoloConvLayer[ 512, 1, 1, 19 ],
   "layer76"->YoloConvLayer[ 1024, 3, 1, 19 ],
   "layer77"->YoloConvLayer[ 512, 1, 1, 19 ],
   "layer78"->YoloConvLayer[ 1024, 3, 1, 19 ],
   "layer79"->YoloConvLayer[ 512, 1, 1, 19 ],
   "layer80"->YoloConvLayer[ 1024, 3, 1, 19 ],
   "layer81"->ConvolutionLayer[ 1818, {1,1} ],
   "layer84"->YoloConvLayer[ 256, 1, 1, 19],
   "layer85"->DeconvolutionLayer[ 256, {2,2}, "Stride"->2],
   "layer86"->CatenateLayer[],
   "layer87"->YoloConvLayer[ 256, 1, 1, 38 ],
   "layer88"->YoloConvLayer[ 512, 3, 1, 38 ],
   "layer89"->YoloConvLayer[ 256, 1, 1, 38 ],
   "layer90"->YoloConvLayer[ 512, 3, 1, 38 ],
   "layer91"->YoloConvLayer[ 256, 1, 1, 38 ],
   "layer92"->YoloConvLayer[ 512, 3, 1, 38 ],
   "layer93"->ConvolutionLayer[ 1818, {1,1} ],
   "layer96"->YoloConvLayer[ 128, 1, 1, 38],
   "layer97"->DeconvolutionLayer[ 128, {2,2} ,"Stride"->2],
   "layer98"->CatenateLayer[],
   "layer99"->YoloConvLayer[  128, 1, 1, 76 ],
   "layer100"->YoloConvLayer[ 256, 3, 1, 76 ],
   "layer101"->YoloConvLayer[ 128, 1, 1, 76 ],
   "layer102"->YoloConvLayer[ 256, 3, 1, 76 ],
   "layer103"->YoloConvLayer[ 128, 1, 1, 76 ],
   "layer104"->YoloConvLayer[ 256, 3, 1, 76 ],
   "layer105"->ConvolutionLayer[ 1818, {1,1} ]
}
,
{
   "layer0"->"layer4"->"layer8"->"layer11"->"layer15"->"layer18"->"layer21"->"layer24"->"layer27"->
   "layer30"->"layer33"->"layer36"->"layer40"->"layer43"->"layer46"->"layer49"->"layer52"->"layer55"->"layer58"->
   "layer61"->"layer65"->"layer68"->"layer71"->"layer74"->"layer75"->"layer76"->"layer77"->"layer78"->"layer79"->"layer80"->"layer81",
   "layer79"->"layer84"->"layer85",
   {"layer85","layer61"}->"layer86"->"layer87"->"layer88"->"layer89"->"layer90"->"layer91"->"layer92"->"layer93",
   "layer91"->"layer96"->"layer97",
   {"layer97","layer36"}->"layer98"->"layer99"->"layer100"->"layer101"->"layer102"->"layer103"->"layer104"->"layer105",
   "layer81"->NetPort["Layer81"],"layer93"->NetPort["Layer93"],"layer105"->NetPort["Layer105"]
}];


multiboxObjDecoder[ anchors_, width_, height_ ] := NetChain[{ReshapeLayer[{anchors,606,width,height}],PartLayer[{All,5}],LogisticSigmoid}]


multiboxClassesDecoder[ anchors_, width_, height_ ] := NetChain[{ReshapeLayer[{anchors,606,width,height}],PartLayer[{All,6;;606}],LogisticSigmoid}]


multiboxLocationsDecoder[ layerNo_, anchors_, width_, height_ ] := NetGraph[{
   "reshape"->ReshapeLayer[{anchors,606,width,height}],
   "cx"->{ PartLayer[{All,1}],LogisticSigmoid,ConstantPlusLayer[], ElementwiseLayer[608*#/width&] },
   "cy"->{ PartLayer[{All,2}],LogisticSigmoid,ConstantPlusLayer[], ElementwiseLayer[608*(1-#/height)&] },
   "width"->{ PartLayer[{All,3}], ElementwiseLayer[ Exp ], ConstantTimesLayer[] },
   "height"->{ PartLayer[{All,4}], ElementwiseLayer[ Exp ], ConstantTimesLayer[] },
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
