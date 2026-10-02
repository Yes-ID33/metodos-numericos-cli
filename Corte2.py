import os
import subprocess
import numpy as np
import pandas as pd
from sympy import Symbol, lambdify, sympify, Interval, pi, E, expand, simplify, diff
from sympy.calculus.util import continuous_domain

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda x: "%.8f" % x)

np.seterr(all="ignore")

def limpiar_pantalla():
    subprocess.run("cls" if os.name == "nt" else "clear", shell=True)

def creditos():
    print("\nEste programa fue realizado por:")
    print("Jordan Corrales Murillo")
    print("Jorge Andrés Grisales Herrera")
    print("Pablo Andrés Rendón Correa")
    print("y, Yesid Maldonado Carvajal")
    print("¡Gracias por preferirnos!")
    input("\nPresiona ENTER para regresar al menú principal...")

def manual():
    limpiar_pantalla()
    print("=" * 60)
    print("        GUÍA DE SINTAXIS Y OPERADORES MATEMÁTICOS")
    print("=" * 60)
    print("Suma / Resta:       +  y  -        Ejemplo: x + 5")
    print("Multiplicación:     *              Ejemplo: 3*x (¡Obligatorio el '*'!)")
    print("División:           /              Ejemplo: (x + 1)/(x - 2)")
    print("Potencia:           **             Ejemplo: x**3 (para x³)")
    print("Raíz cuadrada:      sqrt(x)        Ejemplo: sqrt(x + 2)")
    print("Logaritmo natural:  log(x)         Ejemplo: log(x)")
    print("Logaritmo base b:   log(x, b)      Ejemplo: log(3x, 5)")
    print("Exponencial (e^x):  exp(x)         Ejemplo: exp(-14x)")
    print("Trigonométricas:    sin(x), cos(x), tan(x), asin(x), acos(x), atan(x)")
    print("Constantes:         pi   (número π);   e   (número de Euler, minúscula)")
    print("                    OJO: la 'E' MAYÚSCULA queda libre para usarla como parámetro.")
    print("Parámetros:         puedes usar letras como A, B, C, E, r, L, D, beta1, etc.")
    print("                    El programa las detecta solas y te pedirá su valor")
    print("                    numérico antes de operar (útil para fórmulas físicas).")
    print("Nodos / Puntos:     Puedes ingresar expresiones como 'pi/2', 'sqrt(2)/2', '1/3'.")
    print("=" * 60)
    input("\nPresiona ENTER para regresar al menú principal...")

def pedir_float(mensaje):
    local_dict = {"pi": pi, "e": E}
    while True:
        txt = input(mensaje).strip()
        try:
            val = float(sympify(txt, locals=local_dict))
            return val
        except Exception:
            print("❌ Error: Ingresaste una expresión o valor numérico inválido. Intenta de nuevo.")

def pedir_entero_positivo(mensaje, minimo=2):
    while True:
        try:
            val = int(input(mensaje).strip())
            if val >= minimo:
                return val
            print(f"❌ Error: Debes ingresar un número entero mayor o igual a {minimo}.")
        except ValueError:
            print("❌ Error: Ingresaste un valor inválido. Debe ser un número entero.")

def pedir_parametros(expr):
    x = Symbol("x")
    libres = sorted((expr.free_symbols - {x}), key=lambda s: s.name)
    if not libres:
        return expr
    print(f"\nLa expresión usa los siguientes parámetros: {', '.join(str(s) for s in libres)}")
    valores = {}
    for s in libres:
        valores[s] = pedir_float(f"  Valor de {s}: ")
    return expr.subs(valores)

def pedir_funcion(nombre="f(x)"):
    opc = input("¿Deseas ver el manual de sintaxis primero? (s/n): ")
    if opc.lower() == "s":
        manual()

    x = Symbol("x")
    local_dict = {"pi": pi, "e": E, "E": Symbol("E")}

    while True:
        fn_str = input(f"\nIngresa la función {nombre}: ").strip()
        try:
            expr_raw = sympify(fn_str, locals=local_dict)
            expr = pedir_parametros(expr_raw)
            fn = lambdify(x, expr, modules=["numpy"])

            try:
                float(fn(1.0))
            except Exception as e_test:
                print(f"❌ Error al evaluar la función numéricamente. ¿Usaste otra variable además de 'x' o escribiste mal una constante? (Detalle: {e_test})")
                continue

            return expr, fn
        except Exception as e:
            print(f"❌ Error al interpretar la función: {e}. Intenta de nuevo.")

def pedir_nodos_x(cantidad_nodos):
    x_nodos = []
    print(f"\n--- Ingreso de los {cantidad_nodos} nodos (x_i) ---")
    for i in range(cantidad_nodos):
        while True:
            xi = pedir_float(f"Digite el valor de x_{i}: ")
            if xi in x_nodos:
                print(f"❌ Error: El nodo {xi} ya fue ingresado. Los nodos x_i deben ser todos distintos.")
            else:
                x_nodos.append(xi)
                break
    return np.array(x_nodos, dtype=float)

def pedir_puntos_xy():
    cant = pedir_entero_positivo("¿Cuántos puntos (x_i, y_i) tiene este ejercicio?: ", minimo=2)
    x_nodos = []
    y_nodos = []
    print(f"\n--- Ingreso de los {cant} pares de puntos ---")
    for i in range(cant):
        while True:
            xi = pedir_float(f"Digite el valor de x_{i}: ")
            if xi in x_nodos:
                print(f"❌ Error: El nodo {xi} ya fue ingresado. Los nodos x_i deben ser todos distintos.")
            else:
                x_nodos.append(xi)
                break
        yi = pedir_float(f"Digite el valor de y_{i}: ")
        y_nodos.append(yi)
    return np.array(x_nodos, dtype=float), np.array(y_nodos, dtype=float)

def verificar_continuidad(expr, a, b):
    x = Symbol("x")
    intervalo = Interval(a, b)
    try:
        dominio_continuo = continuous_domain(expr, x, intervalo)
        return dominio_continuo == intervalo
    except Exception:
        return True