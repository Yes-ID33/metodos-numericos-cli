import os
import subprocess
import numpy as np
import pandas as pd
from sympy import Symbol, lambdify, sympify, Interval, pi, E
from sympy.calculus.util import continuous_domain

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda x: "%.8f" % x)

np.seterr(all="ignore")  # evita que se impriman warnings de dominio (sqrt/asin, etc.)


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
    print("=" * 60)
    input("\nPresiona ENTER para regresar al menú principal...")


def pedir_parametros(expr):
    """Detecta símbolos libres distintos de 'x' dentro de la expresión (p. ej. A, B, C, r, L)
    y solicita su valor numérico, sustituyéndolos para dejar la expresión solo en función de x."""
    x = Symbol("x")
    libres = sorted((expr.free_symbols - {x}), key=lambda s: s.name)
    if not libres:
        return expr
    print(f"\nLa expresión usa los siguientes parámetros: {', '.join(str(s) for s in libres)}")
    valores = {}
    for s in libres:
        while True:
            try:
                valores[s] = float(input(f"  Valor de {s}: "))
                break
            except ValueError:
                print("  ❌ Valor inválido, ingresa un número.")
    return expr.subs(valores)


def pedir_funcion(nombre="f(x)"):
    opc = input("¿Deseas ver el manual de sintaxis primero? (s/n): ")
    if opc.lower() == "s":
        manual()

    x = Symbol("x")
    # OJO: SymPy reconoce "E" globalmente como el número de Euler sin importar
    # el diccionario de símbolos locales. Como varios problemas del parcial usan
    # "E" como parámetro (A, B, C, E), se fuerza aquí a que "E" sea un símbolo
    # normal. Para el número de Euler usa la "e" minúscula o exp(1).
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
                print(f"❌ Error al evaluar la función numéricamente. ¿Usaste otra variable "
                      f"además de 'x' o escribiste mal una constante? (Detalle: {e_test})")
                continue

            return expr, fn
        except Exception as e:
            print(f"❌ Error al interpretar la función: {e}. Intenta de nuevo.")


def verificar_continuidad(expr, a, b):
    """Comprueba simbólicamente si f(x) es continua en [a, b]."""
    x = Symbol("x")
    intervalo = Interval(a, b)
    try:
        dominio_continuo = continuous_domain(expr, x, intervalo)
        return dominio_continuo == intervalo
    except Exception:
        # Si SymPy no puede determinar la continuidad analíticamente, permite continuar con advertencia
        return True


def verificar_teorema_punto_fijo(expr, fn, a, b):
    """Verifica automapeo g([a,b]) actual en [a,b] y calcula K = max|g'(x)| < 1."""
    x = Symbol("x")
    puntos = np.linspace(a, b, 400)

    # 1. Automapeo: g(x) debe estar dentro de [a, b] para todo x en [a, b]
    valores_g = np.array([fn(p) for p in puntos], dtype=float)
    valores_g = valores_g[np.isfinite(valores_g)]
    if valores_g.size == 0:
        return False, None
    min_g, max_g = np.min(valores_g), np.max(valores_g)
    automapeo = (min_g >= a - 1e-9) and (max_g <= b + 1e-9)

    # 2. Contracción: K = max |g'(x)| < 1
    derivada = expr.diff(x)
    deriv_fn = lambdify(x, derivada, modules=["numpy"])
    valores_deriv = np.array([abs(deriv_fn(p)) for p in puntos], dtype=float)
    valores_deriv = valores_deriv[np.isfinite(valores_deriv)]
    K = float(np.max(valores_deriv)) if valores_deriv.size else None

    return automapeo, K


def verificar_unicidad(expr, a, b):
    """Analiza si f(x) es estrictamente monótona en [a, b] (indicio de raíz única
    cuando f(a) y f(b) tienen signos opuestos)."""
    x = Symbol("x")
    puntos = np.linspace(a, b, 400)
    derivada = expr.diff(x)
    try:
        deriv_fn = lambdify(x, derivada, modules=["numpy"])
        valores = np.array([deriv_fn(p) for p in puntos], dtype=float)
        valores = valores[np.isfinite(valores)]
        if valores.size == 0:
            return None
        positivos = np.all(valores >= -1e-9)
        negativos = np.all(valores <= 1e-9)
        return positivos or negativos
    except Exception:
        return None


def n_biseccion(a, b, TOL):
    """Calcula n >= log2((b - a) / TOL)."""
    return int(np.ceil(np.log((b - a) / TOL) / np.log(2)))


def n_punto_fijo(expr, fn, a, b, P0, TOL):
    """Calcula n >= log( ((1 - K) * TOL) / |P0 - g(P0)| ) / log(K)."""
    x = Symbol("x")
    try:
        derivada = expr.diff(x)
        deriv_fn = lambdify(x, derivada, modules=["numpy"])
        puntos = np.linspace(a, b, 400)
        K = float(np.max(np.abs([deriv_fn(p) for p in puntos])))

        if K >= 1:
            return None, K  # No cumple la condición de contracción K < 1

        g_P0 = float(fn(P0))
        diferencia_inicial = abs(P0 - g_P0)

        if diferencia_inicial == 0:
            return 0, K  # P0 ya es el punto fijo exacto

        numerador = np.log(((1.0 - K) * TOL) / diferencia_inicial)
        denominador = np.log(K)

        n_teorico = int(np.ceil(numerador / denominador))
        return max(1, n_teorico), K
    except Exception:
        return None, None


def detectar_estabilizacion(valores, decimales=10):
    """Devuelve la primera iteración (1-indexada) en la que el valor ya coincide
    con el valor final dentro de 'decimales' decimales (criterio de estabilización)."""
    if not valores:
        return None
    final = valores[-1]
    umbral = 10 ** (-decimales)
    for i, v in enumerate(valores, start=1):
        try:
            if abs(v - final) < umbral:
                return i
        except TypeError:
            continue
    return len(valores)


def exportar_csv(df, nombre_sugerido):
    if df is None or df.empty:
        return
    op = input("¿Deseas exportar esta tabla a un archivo CSV (para Excel)? (s/n): ")
    if op.lower() == "s":
        nombre = input(f"Nombre del archivo [ENTER para '{nombre_sugerido}']: ").strip() or nombre_sugerido
        if not nombre.endswith(".csv"):
            nombre += ".csv"
        try:
            df.to_csv(nombre, index=False)
            print(f"✅ Archivo guardado como: {os.path.abspath(nombre)}")
        except Exception as e:
            print(f"❌ No se pudo guardar el archivo: {e}")


# ============================================================
#                        BISECCIÓN
# ============================================================

def biseccion(fn, a, b, No, TOL):
    try:
        FA = fn(a)
        FB = fn(b)
    except Exception as e:
        print(f"❌ Error al evaluar f en los extremos del intervalo: {e}")
        return None

    if FA * FB >= 0:
        print(f"f(a):{FA:.6f} y f(b):{FB:.6f} no tienen signos opuestos. "
              f"No se puede garantizar una raíz en [{a}, {b}] con este método.")
        return None

    iteraciones = []
    i = 1

    while i <= No:
        P = (a + b) / 2.0
        try:
            FP = fn(P)
        except Exception as e:
            print(f"⚠️  Error al evaluar en la iteración {i}: {e}")
            break
        error = abs((b - a) / 2.0)

        iteraciones.append({
            "Iteracion": i,
            "a": a,
            "b": b,
            "c": P,
            "f(c)": FP,
            "Error": error,
        })

        if abs(FP) < TOL or error < TOL:
            break

        if FA * FP > 0:
            a = P
            FA = FP
        else:
            b = P

        i += 1

    return pd.DataFrame(iteraciones)


def ejecucion_biseccion():
    limpiar_pantalla()
    print("====--- MÉTODO DE LA BISECCIÓN ---====")

    expr, fn = pedir_funcion()

    try:
        a = float(input("\nIngresa el límite inferior (a): "))
        b = float(input("Ingresa el límite superior (b): "))
        if a >= b:
            print("'a' debe ser estrictamente menor que 'b'.")
            input("\nPresiona ENTER para regresar...")
            return

        TOL = float(input("Tolerancia (TOL) [Presione ENTER para 0.001]: ") or 0.001)
        No = int(input("Máximo de iteraciones (No) [Presione ENTER para 100]: ") or 100)
    except ValueError:
        print("Error: Ingresaste un parámetro inválido")
        input("\nPresiona ENTER para regresar...")
        return

    # a. ¿Es posible aplicar bisección en el intervalo?
    print("\na. ¿Es posible aplicar el método en el intervalo dado?")
    print("   Verificando la continuidad de f(x) en el intervalo...")
    if not verificar_continuidad(expr, a, b):
        print(f"   ❌ La función no es continua en [{a}, {b}]. No se puede aplicar bisección.")
        input("\nPresiona ENTER para regresar...")
        return
    print("   ✅ La función es continua en el intervalo.")

    try:
        fa, fb = fn(a), fn(b)
        print(f"   f({a}) = {fa:.6f}   f({b}) = {fb:.6f}")
        if fa * fb < 0:
            print("   ✅ f(a) y f(b) tienen signos opuestos: SÍ es posible aplicar bisección.")
        else:
            print("   ⚠️ f(a) y f(b) NO tienen signos opuestos: el método podría no converger a una raíz.")
    except Exception as e:
        print(f"   ⚠️ No se pudo evaluar f en los extremos: {e}")

    print("\n   Verificando unicidad de la raíz (monotonía de f en el intervalo)...")
    es_monotona = verificar_unicidad(expr, a, b)
    if es_monotona is True:
        print("   ✅ f(x) es monótona en [a, b]: si hay cambio de signo, la raíz es única.")
    elif es_monotona is False:
        print("   ⚠️ f(x) NO es monótona en [a, b]: podría existir más de una raíz en el intervalo.")
    else:
        print("   ⚠️ No se pudo determinar la monotonía de f(x) analíticamente.")

    # b. Iteraciones teóricas
    n_teorico = n_biseccion(a, b, TOL)
    print(f"\nb. Iteraciones teóricas necesarias para TOL = {TOL:g}: n >= {n_teorico}")

    # c. Iteraciones reales (prueba de escritorio / programa)
    print("\nc. Ejecutando las iteraciones...")
    df_resultado = biseccion(fn, a, b, No, TOL)

    if df_resultado is not None and not df_resultado.empty:
        print("\n TABLA DE ITERACIONES:")
        print(df_resultado.to_string(index=False))
        print(f"\nRaíz aproximada: {df_resultado.iloc[-1]['c']:.8f}")
        print(f"Iteraciones reales utilizadas: {len(df_resultado)}")
        exportar_csv(df_resultado, "biseccion.csv")

    input("\nPresiona ENTER para volver al menú principal...")


# ============================================================
#                        PUNTO FIJO
# ============================================================

def punto_fijo(fn, x0, TOL, No, K=None):
    """
    Método de punto fijo para resolver x = g(x).

    Si se provee K (constante de contracción, 0 < K < 1), además del criterio
    clásico |Pn - Pn-1| <= TOL, se calcula la cota teórica de Ostrowski:
        K^n * |P0 - P1| / (1 - K) <= TOL
    y se reporta en qué iteración se cumple cada criterio.
    """
    iteraciones = []
    x_ant = x0
    P0 = x0
    try:
        P1 = fn(P0)
    except Exception:
        P1 = None

    i = 1
    n_tol = None
    n_ostrowski = None

    while i <= No:
        try:
            x_actual = fn(x_ant)
        except Exception as e:
            print(f"⚠️  Error al evaluar en la iteración {i}: {e}")
            break

        error = abs(x_actual - x_ant)
        fila = {"Iteracion": i, "x_anterior": x_ant, "x_actual": x_actual, "Error": error}

        if K is not None and 0 < K < 1 and P1 is not None:
            cota = (K ** i) * abs(P0 - P1) / (1 - K)
            fila["Cota_Ostrowski"] = cota
            if n_ostrowski is None and cota <= TOL:
                n_ostrowski = i

        iteraciones.append(fila)

        if n_tol is None and error < TOL:
            n_tol = i

        if error < TOL:
            break

        x_ant = x_actual
        i += 1

    return pd.DataFrame(iteraciones), n_tol, n_ostrowski


def ejecucion_punto_fijo():
    limpiar_pantalla()
    print("====--- MÉTODO DE PUNTO FIJO ---====")
    print("Recuerda: la función debe estar despejada en la forma x = g(x)\n")

    expr, fn = pedir_funcion("g(x)  (despejada de x = g(x))")

    try:
        a = float(input("\nIngresa el límite inferior del intervalo (a): "))
        b = float(input("Ingresa el límite superior del intervalo (b): "))
        if a >= b:
            print("Error: 'a' debe ser estrictamente menor que 'b'.")
            input("\nPresiona ENTER para regresar...")
            return

        x0 = float(input(f"Ingresa el valor inicial (x0) dentro de [{a}, {b}]: "))
        if not (a <= x0 <= b):
            print(f"Advertencia: x0 ({x0}) está fuera del intervalo [{a}, {b}].")

        TOL = float(input("Tolerancia (TOL) [Presione ENTER para 0.001]: ") or 0.001)
        No = int(input("Máximo de iteraciones (No) [Presione ENTER para 100]: ") or 100)
    except ValueError:
        print("Error: Ingresaste un parámetro numérico inválido.")
        input("\nPresiona ENTER para regresar...")
        return

    # 1. Continuidad
    print("\n1. Verificando la continuidad de g(x) en el intervalo...")
    if not verificar_continuidad(expr, a, b):
        print(f"   ❌ g(x) NO es continua en [{a}, {b}].")
        input("\nPresiona ENTER para regresar...")
        return
    print("   ✅ g(x) es continua en el intervalo.")

    # 2. Teorema de punto fijo (Automapeo y K < 1)
    print("\n2. Verificando el Teorema del Punto Fijo (Automapeo y K < 1)...")
    automapeo, K = verificar_teorema_punto_fijo(expr, fn, a, b)

    if automapeo:
        print("   ✅ Automapeo confirmado: g([a, b]) está contenido en [a, b].")
    else:
        print("   ⚠️ Advertencia: g(x) se sale del intervalo [a, b] en algunos puntos (no hay automapeo).")

    if K is not None:
        print(f"   Constante de contracción K = max|g'(x)| en [{a}, {b}]: {K:.6f}")
        if K < 1:
            print("   ✅ Se cumple K < 1 → existe punto fijo único en [a, b] (Teorema de Banach).")
        else:
            print("   ⚠️ Advertencia: K >= 1. El método podría divergir u oscilar.")
    else:
        print("   ⚠️ No se pudo calcular K.")

    if not automapeo or (K is not None and K >= 1):
        seguir = input("\n¿Deseas continuar con las iteraciones de todas formas? (s/n): ")
        if seguir.lower() != "s":
            input("\nPresiona ENTER para regresar al menú principal...")
            return

    # 3. Iteraciones teóricas
    n_teorico, _ = n_punto_fijo(expr, fn, a, b, x0, TOL)
    if n_teorico is not None:
        print(f"\n3. Iteraciones teóricas estimadas (n): {n_teorico}")
    else:
        print("\n3. No se pudo determinar n teórico (K >= 1 o error numérico).")

    # 4. Ejecución del algoritmo (con doble criterio de tolerancia)
    df_resultado, n_tol, n_ostrowski = punto_fijo(fn, x0, TOL, No, K=K)

    if df_resultado is not None and not df_resultado.empty:
        print("\n4. TABLA DE ITERACIONES:")
        print(df_resultado.to_string(index=False))
        print(f"\nRaíz aproximada: {df_resultado.iloc[-1]['x_actual']:.10f}")

        print(f"\nIteraciones necesarias con |Pn - Pn-1| <= TOL: {n_tol}")
        if K is None or K >= 1:
            print("La cota k^n|P0-P1|/(1-k) no aplica porque no se cumple K < 1.")
        elif n_ostrowski is not None:
            print(f"Iteraciones necesarias con la cota k^n|P0-P1|/(1-k) <= TOL: {n_ostrowski}")
        else:
            print("La cota k^n|P0-P1|/(1-k) no bajó de TOL dentro de las iteraciones ejecutadas "
                  "(es una cota conservadora: puede exigir más iteraciones que el criterio |Pn-Pn-1|).")

        n_estab = detectar_estabilizacion(list(df_resultado["x_actual"]))
        print(f"La solución se estabiliza aproximadamente en la iteración {n_estab} "
              f"(valor: {df_resultado.iloc[n_estab - 1]['x_actual']:.10f}).")

        exportar_csv(df_resultado, "punto_fijo.csv")

    input("\nPresiona ENTER para volver al menú principal...")


# ============================================================
#                     NEWTON - RAPHSON
# ============================================================

def newton(expr, fn, x0, TOL, No):
    """Método de Newton-Raphson para encontrar la raíz de f(x) = 0."""
    x = Symbol("x")
    derivada = expr.diff(x)
    deriv_fn = lambdify(x, derivada, modules=["numpy"])

    iteraciones = []
    i = 1

    while i <= No:
        try:
            f_x0 = fn(x0)
            df_x0 = deriv_fn(x0)
        except Exception as e:
            print(f"⚠️  Error al evaluar en la iteración {i}: {e}")
            break

        if df_x0 == 0:
            print(f"Error: La derivada se hizo cero en la iteración {i} (x = {x0}).")
            break

        x_actual = x0 - (f_x0 / df_x0)
        error = abs(x_actual - x0)

        try:
            f_actual = fn(x_actual)
        except Exception:
            f_actual = float("nan")

        iteraciones.append({
            "Iteracion": i,
            "x0": x0,
            "x_actual": x_actual,
            "f(x_actual)": f_actual,
            "Error": error
        })

        if error < TOL:
            break

        i += 1
        x0 = x_actual

    return pd.DataFrame(iteraciones)


def ejecucion_newton():
    limpiar_pantalla()
    print("====--- MÉTODO DE NEWTON-RAPHSON ---====")

    expr, fn = pedir_funcion()

    try:
        x0 = float(input("\nIngresa el valor inicial (X0): "))
        TOL = float(input("Tolerancia (TOL) [Presione ENTER para 0.001]: ") or 0.001)
        No = int(input("Máximo de iteraciones (No) [Presione ENTER para 100]: ") or 100)
    except ValueError:
        print("Error: Ingresaste un parámetro numérico inválido.")
        input("\nPresiona ENTER para regresar...")
        return

    df_resultado = newton(expr, fn, x0, TOL, No)

    if df_resultado is not None and not df_resultado.empty:
        print("\n TABLA DE ITERACIONES:")
        print(df_resultado.to_string(index=False))
        print(f"\nRaíz aproximada: {df_resultado.iloc[-1]['x_actual']:.10f}")
        print(f"Iteraciones necesarias para TOL = {TOL:g}: {len(df_resultado)}")

        n_estab = detectar_estabilizacion(list(df_resultado["x_actual"]))
        print(f"La solución se estabiliza aproximadamente en la iteración {n_estab} "
              f"(valor: {df_resultado.iloc[n_estab - 1]['x_actual']:.10f}).")

        exactos = df_resultado[df_resultado["Error"] == 0.0]
        if not exactos.empty:
            print(f"Se alcanza la solución 'exacta' (error = 0 en precisión de máquina) "
                  f"en la iteración {int(exactos.iloc[0]['Iteracion'])}.")
        else:
            print("No se alcanzó un error exactamente igual a 0 dentro del número de iteraciones dado.")

        exportar_csv(df_resultado, "newton.csv")

    input("\nPresiona ENTER para volver al menú principal...")


# ============================================================
#        CLASIFICACIÓN DE PUNTOS CRÍTICOS (MÁXIMO / MÍNIMO)
# ============================================================

def clasificar_punto_critico():
    """
    Útil para problemas de optimización: dada la función original F(x) y un
    punto crítico x0 (obtenido previamente con bisección, punto fijo o Newton
    aplicado a F'(x) = 0), determina si en x0 hay un máximo o un mínimo local
    usando el criterio de la segunda derivada.
    """
    limpiar_pantalla()
    print("====--- CLASIFICACIÓN DE PUNTOS CRÍTICOS (MÁXIMO / MÍNIMO) ---====")
    print("Ingresa la función ORIGINAL F(x) (no su derivada).")
    expr, fn = pedir_funcion("F(x)")

    try:
        x0 = float(input("\nIngresa el valor del punto crítico x0 (donde F'(x0) ≈ 0): "))
    except ValueError:
        print("Valor inválido.")
        input("\nPresiona ENTER para regresar...")
        return

    x = Symbol("x")
    d1 = expr.diff(x)
    d2 = expr.diff(x, 2)
    d1_fn = lambdify(x, d1, modules=["numpy"])
    d2_fn = lambdify(x, d2, modules=["numpy"])

    try:
        val_d1 = float(d1_fn(x0))
        val_d2 = float(d2_fn(x0))
        val_f = float(fn(x0))
    except Exception as e:
        print(f"❌ Error al evaluar las derivadas en x0: {e}")
        input("\nPresiona ENTER para regresar...")
        return

    print(f"\nF'(x0)  = {val_d1:.8f}   (debería ser cercano a 0 para que x0 sea crítico)")
    print(f"F''(x0) = {val_d2:.8f}")

    if abs(val_d1) > 1e-3:
        print("⚠️  Advertencia: F'(x0) no está cercano a 0; x0 podría no ser un punto crítico real.")

    if val_d2 > 0:
        print(f"\n➡️  En x0 = {x0:.8f} la función tiene un MÍNIMO local. F(x0) = {val_f:.8f}")
    elif val_d2 < 0:
        print(f"\n➡️  En x0 = {x0:.8f} la función tiene un MÁXIMO local. F(x0) = {val_f:.8f}")
    else:
        print("\n➡️  F''(x0) = 0: el criterio de la segunda derivada no es concluyente "
              "(podría ser un punto de inflexión).")

    input("\nPresiona ENTER para volver al menú principal...")


# ============================================================
#                           MENÚ
# ============================================================

def menu_inicio():
    while True:
        limpiar_pantalla()
        print("=== CALCULADORA DE MÉTODOS NUMÉRICOS ===")
        print("1. Método de la Bisección")
        print("2. Método de Punto Fijo")
        print("3. Método de Newton-Raphson")
        print("4. Clasificar punto crítico (máximo/mínimo, para optimización)")
        print("5. Ver manual de sintaxis")
        print("6. Ver créditos")
        print("7. Salir")

        opc = input("\nSelecciona una opción (1-7): ")

        if opc == "1":
            ejecucion_biseccion()
        elif opc == "2":
            ejecucion_punto_fijo()
        elif opc == "3":
            ejecucion_newton()
        elif opc == "4":
            clasificar_punto_critico()
        elif opc == "5":
            manual()
        elif opc == "6":
            creditos()    
        elif opc == "7":
            print("¡Hasta luego!")
            break


if __name__ == "__main__":
    menu_inicio()
