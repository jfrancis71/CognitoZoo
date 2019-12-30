(* ::Package:: *)

(* Note, in practice it can sort of work but prone to numerical instability, especially the
   2D model
*)

<<"Experimental/GenerativeModels/CZGenerativeUtils.m"


gauss = NetChain[{ElementwiseLayer[-0.5*#^2 - Log[Sqrt[2*Pi]]&]},"Input"->"Scalar"];


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


planar[ next_, nextConditional_:False ] := NetGraph[{
   "parameters"->PartLayer[1;;3],
   If[ nextConditional, "nextParameters"->PartLayer[4;;], Nothing ],
   "scalingw"->PartLayer[1;;1],
   "scalingu"->PartLayer[2;;2],
   "bias"->PartLayer[3;;3],
   
   "scalinguprime"->scalinguprime,
   "t1"->ThreadingLayer[Times],
   "expr1"->ThreadingLayer[Plus], (* w x + b *)
   "tanh"->ElementwiseLayer[Tanh],
   "t2"->ThreadingLayer[Times],
   "expr2"->ThreadingLayer[Plus], (* x + u tanh( w x + b ) *)
   "next"->next,
   "sech"->sechlayer,
   "sq"->ElementwiseLayer[#^2&],
   "t3"->ThreadingLayer[Times],
   "expr3"->ElementwiseLayer[(1+#)&], (* 1 + u w Sech[ w x + b ]^2 *)
   "logderiv"->ElementwiseLayer[Log[#+10^-3]&],
   "t4"->ThreadingLayer[Plus]},{
   NetPort["Conditional"]->"parameters"->{"scalingw","scalingu","bias"},
   If[ nextConditional, NetPort["Conditional"]->"nextParameters"->NetPort[ {"next","Conditional"} ], Nothing ],
   {"scalingw",NetPort["Input"]}->"t1",
   {"bias","t1"}->"expr1"->"tanh",
   "scalingw"->NetPort[{"scalinguprime","Scalingw"}],
   "scalingu"->NetPort[{"scalinguprime","Scalingu"}],
   {"scalinguprime","tanh"}->"t2",
   {"t2",NetPort["Input"]}->"expr2"->"next",
   "expr1"->"sech"->"sq",
   {"sq","scalinguprime","scalingw"}->"t3"->"expr3"->"logderiv",
   {"next","logderiv"}->"t4"
}];


l1 = NetGraph[{
   "conditional"->ConstantArrayLayer[{6}],
   "planar"->planar[ planar[gauss, False], True ],
   (*"planar"->planar[ gauss, False ],*)
   "loss"->ElementwiseLayer[-#&]},{
   "conditional"->NetPort[{"planar","Conditional"}],
   "planar"->"loss",
   "loss"->NetPort["Loss"],
   "loss"->NetPort["Dummy"]},"Input"->"Scalar","Loss"->"Scalar"];
(* Our dummy port is in their because other parts of this framework will pass in "Input" and ask for "Loss"
   but if there is only one output port, it is confused as to the meaning of port "Loss". So I just want a second
   output port.
*)


l2init = NetReplacePart[NetInitialize@l1,
   {"conditional","Array"}->Join[Append[RandomReal[{0,1}-.5,2],0.],Append[RandomReal[{0,1}-.5,2],0.]] ];


l2init = NetReplacePart[NetInitialize@l1,
   {"conditional","Array"}->Join[Append[RandomReal[{0,1}-.5,2],0.]] ];


SyntaxInformation[ CZNormFlowModel ]= {"ArgumentsPattern"->{}};
SyntaxInformation[ CZRealVariable ]= {"ArgumentsPattern"->{}};


CZCreateNormFlowRealVariable[] := CZGenerativeModel[ CZNormFlowModel, CZRealVariable[], Identity, l2init ];


l12D = NetGraph[{
   "p1"->PartLayer[1;;1],
   "p2"->PartLayer[2;;2],
   "conditional1"->ConstantArrayLayer[{6}],
   "conditional2"->{8,Ramp,8,Ramp,6},
   "planar1"->planar[ planar[gauss, False], True ],
   "planar2"->planar[ planar[gauss, False], True ],
   "loss1"->ElementwiseLayer[-#&],
   "loss2"->ElementwiseLayer[-#&],
   "loss"->TotalLayer[]},{
   "p1"->NetPort["planar1","Input"],
   "p2"->NetPort["planar2","Input"],
   "conditional1"->NetPort[{"planar1","Conditional"}],
   "p1"->"conditional2"->NetPort[{"planar2","Conditional"}],
   "planar1"->"loss1",
   "planar2"->"loss2",
   {"loss1","loss2"}->"loss",
   "loss"->NetPort["Loss"],
   "loss1"->NetPort["Dummy"]},"Input"->{2},"Loss"->"Scalar"];
(* Our dummy port is in their because other parts of this framework will pass in "Input" and ask for "Loss"
   but if there is only one output port, it is confused as to the meaning of port "Loss". So I just want a second
   output port.
*)


l22Dinit = NetReplacePart[NetInitialize@l12D,
   {"conditional1","Array"}->Join[Append[RandomReal[{0,1}-.5,2],0.],Append[RandomReal[{0,1}-.5,2],0.]] ];


(*
   Method\[Rule]{"ADAM","GradientClipping"\[Rule].1}
   Training is not hugely stable, have used above NetTrain option with some modest success.
*)


CZCreateNormFlowReal2DVariable[] := CZGenerativeModel[ CZNormFlowModel, CZRealVector[2], Identity, l22Dinit ];
