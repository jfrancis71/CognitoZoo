(* ::Package:: *)

<<"Experimental/GenerativeModels/CZVariationalAutoencoders.m"
<<"Experimental/GenerativeModels/CZPixelCNN.m"


PixelVaEEncoderBinaryImage[ imageDims_, latentUnits_ ] := NetChain[{
   FlattenLayer[],
   CZCreateEncoder[ imageDims[[1]]*imageDims[[2]], latentUnits ]}];


PixelVaEEncoderDiscreteImage[ imageDims_, latentUnits_ ] := NetChain[{
   FlattenLayer[],
   CZCreateEncoder[ imageDims[[1]]*imageDims[[2]]*10, latentUnits ]}];


CZPixelVaEDecoder[ crossEntropyType_, imageDims_ ] := NetGraph[{
   {500,Ramp},
   {500,Ramp},
   {imageDims[[1]]*imageDims[[2]]},
   ReshapeLayer[imageDims],
   CZCreatePixelCNNConditionalNet[ crossEntropyType, PixelCNNOrdering[ imageDims ] ]},{
   NetPort["Conditional"]->1->2->3->4->NetPort[{5,"Conditional"}],
   NetPort["Target"]->NetPort[{5,"Image"}]}];


CZCreatePixelVaEBinary[ imageDims_:{28,28}, latentUnits_:8 ] := CZGenerativeModel[
   CZPixelVaE[ latentUnits ], 
   CZBinary[imageDims],
   Identity,
   CZCreateVaENet[ PixelVaEEncoderBinaryImage[ imageDims, latentUnits ], CZPixelVaEDecoder[ "Binary", imageDims ] ]];


e1 = PixelVaEEncoderBinaryImage[ {28,28}, 8 ]


e2 = CZPixelVaEDecoder[ "Binary", {28,28} ]


CZCreateVaENet[ e1, e2 ]


CZCreatePixelVaEDiscreteImage[ imageDims_:{28,28}, latentUnits_:8 ] := CZGenerativeModel[
   CZPixelVaE[ latentUnits ], 
   CZDiscreteImage[imageDims],
   CZOneHot,
   CZCreateVaENet[ PixelVaEEncoderDiscreteImage[ imageDims, latentUnits ], CZPixelVaEDecoder[ "Probabilities", imageDims ] ]];


SyntaxInformation[ CZPixelVaE ]= {"ArgumentsPattern"->{_}};


CZSample[ CZGenerativeModel[ CZPixelVaE[ latentUnits_ ], inputType_, encoder_, pixelCNNNet_ ] ] := (
   z = CZSampleVaELatent[ latentUnits ];
   cond = NetTake[ NetExtract[ pixelCNNNet, "decoder" ],{1,4} ][ z ];
   CZSampleConditionalPixelCNN[ NetExtract[ pixelCNNNet, {"decoder",5} ], inputType, encoder, cond ]
)
