import numpy as np
class regression():

    def __init__(self, x, y, z, X, Y, df):
        self.x = x
        self.y = y
        self.z = z
        self.X = X
        self.Y = Y
        self.df = df

    # Linear regression of two dimensions
    def linear(self):

        E = 0
        C = 0
        A = 0
        B = 0
        m = 0
        bi = 0
        p = len(self.x)

        for i in range(len(self.x)):

            B += self.x[i]
            E += self.z[i]
            A += self.x[i] * self.x[i]
            C += self.x[i] * self.z[i]

        m = ( (C * p) -  (E * B) ) / ( (A*p) - (B * B) )
        bi = ( E - m * B ) /  p

        return m , bi



    # Linear regression of three dimensions
    def linear_3D(self):

        a = 0
        b = 0
        g = 0
        e = 0
        t = 0
        la = 0
        u = 0
        r = 0
        th = 0
        z = 0
        m1 = 0
        m2 = 0
        bi = 0
        p = len(self.x)

        for i in range(len(self.x)):

            a += self.x[i] * self.x[i]
            b += self.x[i] * self.y[i]
            g += self.x[i]
            e += self.x[i] * self.z[i]
            th += self.y[i] * self.y[i]
            la += self.y[i]
            u += self.y[i] * self.z[i]
            z += self.z[i]

        t = la
        r = g
        m_list = [[a, b, g], [b, th, la], [r, t, p ] ]
        A = np.array(m_list)
        B = np.array([e, u, z])
        X = np.linalg.inv(A).dot(B)
        m1 = X[0]
        m2 = X[1]
        bi = (z - (t * m2) - (r * m1)) / (p)

        return m1, m2, bi



    # Linear regression for multiple dimensions
    def mlinear(self):

        M_row = np.array([])
        p = len(self.df.index)
        X1_Sum = 0
        X2_Sum = 0
        X11_Sum = 0
        Cx = 0
        M = np.zeros((1, len(self.X.columns) + 1))
        C = np.array([])
        last_row = np.array([])

        Y_Sum = 0

        for i in range(len(self.Y.index)):
            Y_Sum += self.Y.iat[i, 0]

        for j in range(len(self.X.columns)):

            for i in range(len(self.X.columns)):

                # Getting the Sums
                for k in range(len(self.X.index)):
                    X1_Sum += self.X.iat[k, i]
                    X2_Sum += self.X.iat[k, j]
                    X11_Sum += self.X.iat[k, j] * self.X.iat[k, i]
                    if i == len(self.X.columns) - 1:
                        Cx += self.X.iat[k, j] * self.Y.iat[k, 0]

                Coe = X11_Sum

                M_row = np.append(M_row, Coe)  # Adding value of Coe to each M_row

                Coe = 0  # Resetting value of coeffiecient

                if i == len(self.X.columns) - 1:  # Once row loop is done, insert summation of values of corresponding j-dimension into last row
                    last_row = np.append(last_row, X2_Sum)
                    M_row = np.append(M_row, X2_Sum)
                    C = np.append(C, Cx )
                    Cx = 0

                # Reset coeffietients
                X1_Sum = 0
                X2_Sum = 0
                X11_Sum = 0

            M = np.vstack([M, M_row])  # Insert row in M matrix

            M_row = np.array([])  # Reset row value

        last_row = np.append(last_row, p)
        M = np.delete(M, 0, 0)
        M = np.vstack([M, last_row])
        C = np.append(C, Y_Sum)

        A = np.linalg.inv(M).dot(C)

        return A







