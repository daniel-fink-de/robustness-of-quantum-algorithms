OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
U1q(1.5707963267948966, 7.853981405506575) q[0];
rzz(pi/2) q[0], q[2];
U1q(1.5707963267948966, 6.2831850733080845) q[0];
U1q(1.5707963267948966, 4.712388929657958) q[2];
rzz(pi/2) q[0], q[2];
U1q(1.5707963267948966, 12.5663704911164) q[0];
U1q(1.5707963267948966, 6.283185257274085) q[2];
U1q(1.5707963267948966, 6.675884279049406) q[0];
rzz(pi/2) q[0], q[1];
U1q(1.5707963267948966, 9.81747708718499) q[0];
rzz(pi/2) q[0], q[2];
U1q(1.5707963267948966, 6.675884403872177) q[0];
U1q(1.5707963267948966, 7.461282545347018) q[0];
rzz(pi/2) q[0], q[1];
U1q(1.5707963267948966, 2.7488935691145415) q[0];
U1q(1.5707963267948966, 7.068583401939583) q[1];
U1q(1.5707963267948966, 6.283185231791342) q[1];
rzz(pi/2) q[1], q[2];
U1q(3.141592653589793, 5.1050879804723) q[1];
U1q(3.141592653589793, 4.516039295420052) q[2];
U1q(1.5707963267948966, 2.356194477296626) q[1];
U1q(1.5707963267948966, 1.570796326186133) q[2];
