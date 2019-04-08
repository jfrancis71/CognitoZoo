(* ::Package:: *)

hdfFile = "/Users/julian/yolov3/darknet/Yolo9000.hdf5";


YoloConvLayerInitRules[ netName_, hdfName_ ] := {
   {netName,1,"Weights"}->Import[hdfFile,{"Datasets",hdfName<>"_weights"}],
   {netName,2,"Beta"}->Import[hdfFile,{"Datasets",hdfName<>"_biases"}],
   {netName,2,"MovingMean"}-> Import[hdfFile,{"Datasets",hdfName<>"_rolling_mean"}],
   {netName,2,"MovingVariance"}->Import[hdfFile,{"Datasets",hdfName<>"_rolling_variance"}],
   {netName,2,"Gamma"}-> Import[hdfFile,{"Datasets",hdfName<>"_scales"}]
};


SmallBlockInitRules[ name_String, firstLayer_ ] := Map[Prepend[#[[1]],name]->#[[2]]&,Join[
   YoloConvLayerInitRules[ 1, "layer"<>ToString[ firstLayer ] ],
   YoloConvLayerInitRules[ 2, "layer"<>ToString[ firstLayer +1]]
]];


LargeBlockInitRules[ name_String, firstLayer_Integer ] := Map[Prepend[#[[1]],name]->#[[2]]&,Join[
   YoloConvLayerInitRules[ 1, "layer"<>ToString[firstLayer] ],
   YoloConvLayerInitRules[ 2, "layer"<>ToString[firstLayer+1] ],
   YoloConvLayerInitRules[ 3, "layer"<>ToString[firstLayer+2] ]
]];


yoloConvNetInitRules = Join[
   YoloConvLayerInitRules[ "layer0", "layer0"],
   YoloConvLayerInitRules[ "layer2", "layer2"],
   LargeBlockInitRules[ "layer6", 4 ],
   LargeBlockInitRules[ "layer10", 8 ],
   LargeBlockInitRules[ "layer14", 12 ],
   SmallBlockInitRules[ "layer16", 15 ],
   LargeBlockInitRules[ "layer20", 18 ],
   SmallBlockInitRules[ "layer22", 21 ],
   {{"layer23","Weights"}->Import[hdfFile,{"Datasets","layer23_weights"}], {"layer23","Biases"}->Import[hdfFile,{"Datasets","layer23_biases"}]}
];


widthScales = 
   Table[{25,96,295}[[n]],{n,1,3},{17},{17}];


heightScales =
   Table[{37,138,308}[[n]],{n,1,3},{17},{17}];


LocationInitRules = {
   { "height", 3, "Scaling" } -> heightScales,
   { "width", 3, "Scaling" } -> widthScales,
   { "cx", 3, "Biases" } -> Table[j,{3},{i,0,17-1},{j,0,17-1}],
   { "cy", 3, "Biases" } -> Table[i,{3},{i,0,17-1},{j,0,17-1}]
};


yolo9000NetRules = Join[
   Map[ Prepend[#[[1]],"Conv"]->#[[2]]&, yoloConvNetInitRules ],
   Map[ Join[{"Decode",2},#[[1]]]->#[[2]]&, LocationInitRules ]
];


yolo9000Init = NetReplacePart[ yolo9000Net, yolo9000NetRules ];


yolo9000Hierarchy = Import["/Users/julian/yolov3/darknet/data/9k.tree"];


yolo9000Graph = Table[(yolo9000Hierarchy[[k,2]]+1)->k,{k,1,9418}];


yolo9000Names = Import["~/yolov3/darknet/data/9k.names","List"];
