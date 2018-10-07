(* ::Package:: *)

conv1[n_?IntegerQ]:=Sequence[
   ConvolutionLayer[n,1,"PaddingSize"->0,"Stride"->1],
   Ramp
];
conv3[n_?IntegerQ]:=Sequence[
   ConvolutionLayer[n,3,"PaddingSize"->1,"Stride"->1],
   Ramp
];


blockNet4 = NetChain[{
   "conv1a"->{conv3[64]},
   "conv1b"->{conv3[64]},
   "pool1"->PoolingLayer[{2,2},"Stride"->2],
   "conv2a"->{conv3[128]},
   "conv2b"->{conv3[128]},
   "pool2"->PoolingLayer[{2,2},"Stride"->2],
   "conv3a"->{conv3[256]},
   "conv3b"->{conv3[256]},
   "conv3c"->{conv3[256]},
   "pool3"->{PaddingLayer[{{0,0},{0,1},{0,1}}],PoolingLayer[{2,2},"Stride"->2]},
   "conv4a"->{conv3[512]},
   "conv4b"->{conv3[512]},
   "conv4c"->{conv3[512]}
   }];


blockNet7 = NetChain[{
  "pool4"->PoolingLayer[{2,2},"Stride"->2],
   "conv5a"->{conv3[512]},
   "conv5b"->{conv3[512]},
   "conv5c"->{conv3[512]},
   "pool5"->PoolingLayer[{3,3},"Stride"->1,"PaddingSize"->1],
   "conv6"->{ConvolutionLayer[1024,{3,3},"Dilation"->6,"PaddingSize"->6],Ramp},
   "conv7"->{conv1[1024]}
}];


blockNet8 = NetChain[{
   "conv8a"->{conv1[256]},
   "pad2"->PaddingLayer[{{0,0},{1,1},{1,1}}],
   "conv8b"->{ConvolutionLayer[512, {3,3}, "Stride"->2],Ramp}
}];


blockNet9 = NetChain[{
   "conv9a"->{conv1[128]},
   "conv9b"->{ConvolutionLayer[256, {3,3},"Stride"->2,"PaddingSize"->1],Ramp}
}];


blockNet10 = NetChain[{
   "conv10a"->{conv1[128]},
   "conv10b"->{ConvolutionLayer[256, {3,3}],Ramp}
}];


blockNet11 = NetChain[{
   "conv11a"->{conv1[128]},
   "conv11b"->{ConvolutionLayer[256, {3,3}],Ramp}
}];


multiBoxClassesDecoder[ numberOfAnchors_, width_, height_ ] :=
   NetChain[{ ConvolutionLayer[ numberOfAnchors*21, {3,3}, "PaddingSize"->1 ],ReshapeLayer[{numberOfAnchors,21,height,width}], SoftmaxLayer[2] }];


multiBoxLocationDecoder[ numberOfAnchors_, width_, height_ ] :=
   NetGraph[{
      "convloc"->{ConvolutionLayer[ numberOfAnchors*4, {3,3}, "PaddingSize"->1 ]},
      "reshape"->ReshapeLayer[{numberOfAnchors,4,height,width}]
      },{
      "convloc"->"reshape"}];


channelNormalizationNet = NetChain[{
   TransposeLayer[1->3],
   ElementwiseLayer[Log[Max[#^2,10^-20]]&],SoftmaxLayer[],
   ElementwiseLayer[Sqrt],TransposeLayer[1->3],
   ConstantTimesLayer[]
}];


multiBoxLayer1 = NetGraph[{
   "channelNorm1"->channelNormalizationNet,
   "multiboxClasses"->multiBoxClassesDecoder[ 4, 38, 38 ],
   "multiboxLocs"->multiBoxLocationDecoder[ 4, 38, 38 ]},
   {"channelNorm1"->{"multiboxClasses"->NetPort["ClassProb1"],"multiboxLocs"->NetPort["Locs1"]}}];


multiBoxLayer2 = NetGraph[{
   "multiboxClasses"->multiBoxClassesDecoder[ 6, 19, 19 ],
   "multiboxLocs"->multiBoxLocationDecoder[ 6, 19, 19 ]},
   {"multiboxClasses"->NetPort["ClassProb2"],"multiboxLocs"->NetPort["Locs2"]}];


multiBoxLayer3 = NetGraph[{
   "multiboxClasses"->multiBoxClassesDecoder[ 6, 10, 10 ],
   "multiboxLocs"->multiBoxLocationDecoder[ 6, 10, 10 ]},
   {"multiboxClasses"->NetPort["ClassProb3"],"multiboxLocs"->NetPort["Locs3"]}];


multiBoxLayer4 = NetGraph[{
   "multiboxClasses"->multiBoxClassesDecoder[ 6, 5, 5 ],
   "multiboxLocs"->multiBoxLocationDecoder[ 6, 5, 5 ]},
   {"multiboxClasses"->NetPort["ClassProb4"],"multiboxLocs"->NetPort["Locs4"]}];


multiBoxLayer5 = NetGraph[{
   "multiboxClasses"->multiBoxClassesDecoder[ 4, 3, 3 ],
   "multiboxLocs"->multiBoxLocationDecoder[ 4, 3, 3 ]},
   {"multiboxClasses"->NetPort["ClassProb5"],"multiboxLocs"->NetPort["Locs5"]}];


multiBoxLayer6 = NetGraph[{
   "multiboxClasses"->multiBoxClassesDecoder[ 4, 1, 1 ],
   "multiboxLocs"->multiBoxLocationDecoder[ 4, 1, 1 ]},
   {"multiboxClasses"->NetPort["ClassProb6"],"multiboxLocs"->NetPort["Locs6"]}];


ssdConvNet = NetGraph[{
   "blockNet4"->blockNet4,
   "blockNet7"->blockNet7,
   "blockNet8"->blockNet8,
   "blockNet9"->blockNet9,
   "blockNet10"->blockNet10,
   "blockNet11"->blockNet11,
   "multiboxLayer1"->multiBoxLayer1,
   "multiboxLayer2"->multiBoxLayer2,
   "multiboxLayer3"->multiBoxLayer3,
   "multiboxLayer4"->multiBoxLayer4,
   "multiboxLayer5"->multiBoxLayer5,
   "multiboxLayer6"->multiBoxLayer6
   },
   {"blockNet4"->"blockNet7"->"blockNet8"->"blockNet9"->"blockNet10"->"blockNet11",
   "blockNet4"->"multiboxLayer1",
   "blockNet7"->"multiboxLayer2",
   "blockNet8"->"multiboxLayer3",
   "blockNet9"->"multiboxLayer4",
   "blockNet10"->"multiboxLayer5",
   "blockNet11"->"multiboxLayer6"
   }];


multiBoxDecoder[ numberOfAnchors_, width_, height_ ]:= NetGraph[{
      "cx"->{PartLayer[{All,1}],ConstantTimesLayer[],ConstantPlusLayer[],ElementwiseLayer[#*300.&]},
      "cy"->{PartLayer[{All,2}],ConstantTimesLayer[],ConstantPlusLayer[],ElementwiseLayer[(1-#)*300.&]},
      "width"->{PartLayer[{All,3}],ElementwiseLayer[Exp[#*0.2]&],ConstantTimesLayer[],ElementwiseLayer[#*300.&]},
      "height"->{PartLayer[{All,4}],ElementwiseLayer[Exp[#*0.2]&],ConstantTimesLayer[],ElementwiseLayer[#*300.&]},
      "catenate"->CatenateLayer[],(* (C*A)*Y*X *)
      "reshape"->ReshapeLayer[ { 4, numberOfAnchors, height, width } ],
      "transpose"->TransposeLayer[1->2]},{
      {"cx","cy","width","height"}->"catenate"->"reshape"->"transpose"}];


locsToBoxesNet = NetGraph[{
   multiBoxDecoder[ 4, 38, 38 ],
   multiBoxDecoder[ 6, 19, 19 ],
   multiBoxDecoder[ 6, 10, 10 ],
   multiBoxDecoder[ 6, 5, 5 ],
   multiBoxDecoder[ 4, 3, 3 ],
   multiBoxDecoder[ 4, 1, 1 ] },{
   NetPort["Locs1"]->1->NetPort["Boxes1"],NetPort["Locs2"]->2->NetPort["Boxes2"],
   NetPort["Locs3"]->3->NetPort["Boxes3"],NetPort["Locs4"]->4->NetPort["Boxes4"],
   NetPort["Locs5"]->5->NetPort["Boxes5"],NetPort["Locs6"]->6->NetPort["Boxes6"]}];


ReshapeMultiBoxLayer = { TransposeLayer[{3->1,4->2}],FlattenLayer[2]}; 


ssdConcatenationNet = NetGraph[{
   "classProb1"->ReshapeMultiBoxLayer,
   "classProb2"->ReshapeMultiBoxLayer,
   "classProb3"->ReshapeMultiBoxLayer,
   "classProb4"->ReshapeMultiBoxLayer,
   "classProb5"->ReshapeMultiBoxLayer,
   "classProb6"->ReshapeMultiBoxLayer,   
   "classProb"->CatenateLayer[],
   "boxes1"->ReshapeMultiBoxLayer,
   "boxes2"->ReshapeMultiBoxLayer,
   "boxes3"->ReshapeMultiBoxLayer,
   "boxes4"->ReshapeMultiBoxLayer,
   "boxes5"->ReshapeMultiBoxLayer,
   "boxes6"->ReshapeMultiBoxLayer,
   "boxes"->CatenateLayer[]
   },
   {
   NetPort["ClassProb1"]->"classProb1",
   NetPort["ClassProb2"]->"classProb2",
   NetPort["ClassProb3"]->"classProb3",
   NetPort["ClassProb4"]->"classProb4",
   NetPort["ClassProb5"]->"classProb5",
   NetPort["ClassProb6"]->"classProb6",   
   {"classProb1","classProb2","classProb3","classProb4","classProb5","classProb6"}->"classProb"->NetPort["ClassProb"],
   NetPort["Boxes1"]->"boxes1",
   NetPort["Boxes2"]->"boxes2",
   NetPort["Boxes3"]->"boxes3",
   NetPort["Boxes4"]->"boxes4",
   NetPort["Boxes5"]->"boxes5",
   NetPort["Boxes6"]->"boxes6",
   {"boxes1","boxes2","boxes3","boxes4","boxes5","boxes6"}->"boxes"->NetPort["Boxes"]
   }];


ssdNet = NetGraph[{ssdConvNet,locsToBoxesNet,ssdConcatenationNet},
   {
   NetPort[1,"ClassProb1"]->NetPort[3,"ClassProb1"],
   NetPort[1,"ClassProb2"]->NetPort[3,"ClassProb2"],
   NetPort[1,"ClassProb3"]->NetPort[3,"ClassProb3"],
   NetPort[1,"ClassProb4"]->NetPort[3,"ClassProb4"],
   NetPort[1,"ClassProb5"]->NetPort[3,"ClassProb5"],
   NetPort[1,"ClassProb6"]->NetPort[3,"ClassProb6"],
   NetPort[1,"Locs1"]->NetPort[2,"Locs1"],
   NetPort[1,"Locs2"]->NetPort[2,"Locs2"],
   NetPort[1,"Locs3"]->NetPort[2,"Locs3"],
   NetPort[1,"Locs4"]->NetPort[2,"Locs4"],
   NetPort[1,"Locs5"]->NetPort[2,"Locs5"],
   NetPort[1,"Locs6"]->NetPort[2,"Locs6"],
   NetPort[2,"Boxes1"]->NetPort[3,"Boxes1"],
   NetPort[2,"Boxes2"]->NetPort[3,"Boxes2"],
   NetPort[2,"Boxes3"]->NetPort[3,"Boxes3"],
   NetPort[2,"Boxes4"]->NetPort[3,"Boxes4"],
   NetPort[2,"Boxes5"]->NetPort[3,"Boxes5"],
   NetPort[2,"Boxes6"]->NetPort[3,"Boxes6"]
   },
   "Input"->NetEncoder[{"Image",{300,300},"ColorSpace"->"RGB","MeanImage"->{123,117,104}/255.}]];
