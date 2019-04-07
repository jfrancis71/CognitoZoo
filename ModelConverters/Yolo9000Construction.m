(* ::Package:: *)

(* Yolo9000 *)


leakyReLU = ElementwiseLayer[Ramp[#]+Ramp[-#]*-.1&];


(* Don't forget to check the epsilon issue *)


YoloConvLayer[ filters_, filterSize_, stride_, side_ ] :=
   NetChain[{
      ConvolutionLayer[filters,{filterSize,filterSize},"Stride"->stride,"Biases"->ConstantArray[0,filters],"PaddingSize"->If[filterSize==1,0,1]],
      BatchNormalizationLayer["Epsilon"->.00000001,"Input"->{filters,side,side}],
      leakyReLU
}];


(* Residual block filters and resolution unchanged *)
SmallBlock[ inputFilters_, inputSize_ ] := NetChain[{
   YoloConvLayer[ inputFilters/2, 1, 1, inputSize ],
   YoloConvLayer[ inputFilters, 3, 1, inputSize ]}];


LargeBlock[ inputFilters_, inputSize_ ] := NetChain[{
   YoloConvLayer[ inputFilters*2, 3, 1, inputSize ],
   YoloConvLayer[ inputFilters, 1, 1, inputSize ],
   YoloConvLayer[ inputFilters*2, 3, 1, inputSize ] }];


yoloConvNet = NetChain[{
   "layer0"->YoloConvLayer[ 32, 3, 1, 544 ],
   "layer1"->PoolingLayer[ {2,2}, "Stride"->2 ],
   "layer2"->YoloConvLayer[ 64, 3, 1, 272 ],
   "layer3"->PoolingLayer[ {2,2}, "Stride"->2 ],
   "layer6"->LargeBlock[ 64, 136 ],
   "layer7"->PoolingLayer[ {2,2}, "Stride"->2 ],
   "layer10"->LargeBlock[ 128, 68 ],
   "layer11"->PoolingLayer[ {2,2}, "Stride"->2 ],
   "layer14"->LargeBlock[ 256, 34 ],
   "layer16"->SmallBlock[ 512, 34 ],
   "layer17"->PoolingLayer[ {2,2}, "Stride"->2 ],
   "layer20"->LargeBlock[ 512, 17 ],
   "layer22"->SmallBlock[ 1024, 17 ],
   "layer23"->ConvolutionLayer[ 28269, { 1,1 } ],
   "reshape"->ReshapeLayer[ {3, 9423, 17, 17 } ]
}];


multiboxLocationsDecoder = NetGraph[{
   "cx"->{ PartLayer[{All,1}],LogisticSigmoid,ConstantPlusLayer[], ElementwiseLayer[544*#/17&] },
   "cy"->{ PartLayer[{All,2}],LogisticSigmoid,ConstantPlusLayer[], ElementwiseLayer[544*(1-#/17)&] },
   "width"->{ PartLayer[{All,3}], ElementwiseLayer[ Exp ], ConstantTimesLayer[] },
   "height"->{ PartLayer[{All,4}], ElementwiseLayer[ Exp ], ConstantTimesLayer[] },
   "minx"->ThreadingLayer[#1-#2/2&],
   "miny"->ThreadingLayer[#1-#2/2&],
   "maxx"->ThreadingLayer[#1+#2/2&],
   "maxy"->ThreadingLayer[#1+#2/2&],
   "cat"->CatenateLayer[],"reshape1"->ReshapeLayer[{4,3,17,17}]
   },{
   {"cx","width"}->"minx",{"cx","width"}->"maxx",{"cy","height"}->"miny",{"cy","height"}->"maxy",
   {"minx","miny","maxx","maxy"}->"cat"->"reshape1"
}];


decoderNet = NetGraph[{
   PartLayer[ { All, 1;;4, All, All } ],
   multiboxLocationsDecoder,
   PartLayer[ { All, 5 } ],
   PartLayer[ { All, 6;;, All, All } ] },{
   1->2->NetPort["Boxes"],3->NetPort["Objectness"],4->NetPort["ClassHierarchy"]
}];


flatNet = NetGraph[ {
   "locs"->{TransposeLayer[{3->1,4->2,3->4}],FlattenLayer[2]},
   "boxes"->ReshapeLayer[{867,2,2}],
   "ot"->{TransposeLayer[{3->1,1->2}],FlattenLayer[]},
   "ct"->{TransposeLayer[{1->3,2->4}],FlattenLayer[2]}
},{NetPort["Boxes"]->"locs"->"boxes"->NetPort["Boxes"],NetPort["Objectness"]->"ot"->NetPort["Objectness"],NetPort["ClassHierarchy"]->"ct"->NetPort["ClassHierarchy"]}];


yolo9000Net = NetGraph[ { "yoloConvNet"->yoloConvNet, "dec"->decoderNet, "flatNet"->flatNet }, { "yoloConvNet"->"dec",
NetPort["dec","Boxes"]->NetPort["flatNet","Boxes"],
NetPort["dec","Objectness"]->NetPort["flatNet","Objectness"],
NetPort["dec","ClassHierarchy"]->NetPort["flatNet","ClassHierarchy"] },
   "Input"->NetEncoder[{"Image",{544,544},"ColorSpace"->"RGB"}] ];


yolo9000Hierarchy = Import["/Users/julian/yolov3/darknet/data/9k.tree"];


yolo9000Graph = Table[(yolo9000Hierarchy[[k,2]]+1)->k,{k,1,9418}];


yolo9000Names = Import["~/yolov3/darknet/data/9k.names","List"];
