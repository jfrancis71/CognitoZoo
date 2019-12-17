(* ::Package:: *)

<<"Experimental/GenerativeModels/CZGenerativeUtils.m"


gauss = NetChain[{ElementwiseLayer[Exp[-0.5*#^2]/Sqrt[2*Pi]&]},"Input"->"Scalar"];


sechlayer = NetGraph[{
   "numerator"->ElementwiseLayer[2 Exp[#]&],
   "denominator"->ElementwiseLayer[1/(1 + Exp[2 #])&],
   "thread"->ThreadingLayer[Times]},{
   {"numerator","denominator"}->"thread"
}];


scalinguprime = NetGraph[{
   "t1"->ThreadingLayer[Times],
   "m"->ElementwiseLayer[-1+Log[1+Exp[#]]&],
   "inv"->ElementwiseLayer[1/#&],
   "t2"->ThreadingLayer[Times]},{
   {NetPort["Scalingw"],NetPort["Scalingu"]}->"t1"->"m",
   NetPort["Scalingw"]->"inv",
   {"m","inv"}->"t2"}];


planar1[inp_] := NetGraph[{
   "scalingw"->{ConstantArrayLayer[{1}](*,ElementwiseLayer[Exp]*)},
   "scalingu"->{ConstantArrayLayer[{1}](*,ElementwiseLayer[Exp]*)},
   "scalinguprime"->scalinguprime,
   "bias"->ConstantArrayLayer[{1}],
   "t1"->ThreadingLayer[Times],
   "p1"->ThreadingLayer[Plus],
   "tanh"->ElementwiseLayer[Tanh],
   "t2"->ThreadingLayer[Times],
   "p2"->ThreadingLayer[Plus],
   "gauss"->inp,
   "sech"->sechlayer,
   "sq"->ElementwiseLayer[#^2&],
   "t3"->ThreadingLayer[Times],
   "p3"->ElementwiseLayer[(1+#)&],
   "t4"->ThreadingLayer[Times]}(*Invert?*),{
   {"scalingw",NetPort["Input"]}->"t1",
   {"bias","t1"}->"p1"->"tanh",
   "scalingw"->NetPort[{"scalinguprime","Scalingw"}],
   "scalingu"->NetPort[{"scalinguprime","Scalingu"}],
   {"scalinguprime","tanh"}->"t2",
   {"t2",NetPort["Input"]}->"p2"->"gauss",
   "p1"->"sech"->"sq",
   {"sq","scalinguprime","scalingw"}->"t3"->"p3",
   {"gauss","p3"}->"t4"
}];


l1 = NetGraph[{
   "planar"->planar1@planar1[gauss],
   "loss"->ElementwiseLayer[-Log[#]&]},{
   "planar"->"loss",
   "loss"->NetPort["Loss"],
   "loss"->NetPort["Dummy"]},"Input"->"Scalar","Loss"->"Scalar"];
(* Our dummy port is in their because other parts of this framework will pass in "Input" and ask for "Loss"
   but if there is only one output port, it is confused as to the meaning of port "Loss". So I just want a second
   output port.
*)


l2init = NetReplacePart[NetInitialize@l1,
   {{"planar","scalingw",1,"Array"}->Random[],
   {"planar","gauss","scalingw",1,"Array"}->Random[]}];


SyntaxInformation[ CZNormFlowModel ]= {"ArgumentsPattern"->{}};
SyntaxInformation[ CZRealVariable ]= {"ArgumentsPattern"->{}};


CZCreateNormFlowRealVariable[] := CZGenerativeModel[ CZNormFlowModel, CZRealVariable[], Identity, l2init ];
