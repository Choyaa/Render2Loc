import numpy as np

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


mat = np.array([0.872664372374608, 0.481501388383544, 0.0813222366438102, 
                                0.140431548361106, -0.406956456045335,-0.902588180239429,
                               -0.467692071146107,0.776236340213303,-0.42275438580732] )
# mat = mat.reshape(3,3)
# <Rotation>
# 							<M_00>0.872664372374608</M_00>
# 							<M_01>0.481501388383544</M_01>
# 							<M_02>-0.0813222366438102</M_02>
# 							<M_10>0.140431548361106</M_10>
# 							<M_11>-0.406956456045335</M_11>
# 							<M_12>-0.902588180239429</M_12>
# 							<M_20>-0.467692071146107</M_20>
# 							<M_21>0.776236340213303</M_21>
# 							<M_22>-0.42275438580732</M_22>
# 							<Accurate>true</Accurate>
# 						</Rotation>
# 						<Center>
# 							<x>400113.749736579</x>
# 							<y>3138259.76845883</y>
# 							<z>137.26923680678</z>


qvec = rotmat2qvec(mat)
