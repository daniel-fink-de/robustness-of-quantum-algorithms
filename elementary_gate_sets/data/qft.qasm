OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
gate circuitgate_3655043284108571958 (p0, p1, p2) q0, q1, q2 {
	h q2;
	cp(p0) q2, q1;
	h q1;
	cp(p0) q2, q0;
	cp(p0) q1, q0;
	h q0;
	swap q0, q2;
}
circuitgate_3655043284108571958(1.5707963267948966, 0.7853981633974483, 1.5707963267948966) q[0], q[1], q[2];
