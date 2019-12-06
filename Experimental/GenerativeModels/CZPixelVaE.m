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
   {784},
   ReshapeLayer[imageDims],
   CZCreatePixelCNNConditionalNet[ crossEntropyType, PixelCNNOrdering[ imageDims ] ]},{
   NetPort["Conditional"]->1->2->3->4->NetPort[{5,"Conditional"}],
   NetPort["Input"]->NetPort[{5,"Image"}]}];


CZCreatePixelVaEBinaryImage[ imageDims_:{28,28}, latentUnits_:8 ] := CZGenerativeModel[
   CZPixelVaE[ latentUnits ], 
   CZBinaryImage[imageDims],
   Identity,
   CZCreateVaENet[ PixelVaEEncoderBinaryImage[ imageDims, latentUnits ], CZPixelVaEDecoder[ "Binary", imageDims ] ]];


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
