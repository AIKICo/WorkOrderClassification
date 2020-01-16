import numpy as np
import topsis

if __name__ == '__main__':
    top_obj = topsis.Topsis('data.csv')
    w = [0.35, 0.25, 0.25, 0.15]
    im = ["+", "+", "-", "+"]
    decision = top_obj.evaluate(w, im)
    print(decision)
