"""
Лабораторная работа № 9. Вариант 14.
Матрица смежности и списки смежности для ориентированных графов; BFS, DFS;
обнаружение цикла. Разреженные графы (m ≈ 3n), n = 500, 2000, 5000.
Дополнительное задание И7: анализ схемы бизнес-процесса (недостижимые шаги, циклы).
"""
import gc
import json
import random
import statistics
import sys
import time
from collections import deque


# ------------------------------------------------------ представления графа
class MatrixGraph:
    """Ориентированный граф на матрице смежности. Память Θ(n²)."""

    def __init__(self, n):
        self.n = n
        self.matrix = [[0] * n for _ in range(n)]

    def add_edge(self, u, v):
        self.matrix[u][v] = 1                  # дуга u → v

    def has_edge(self, u, v):
        return self.matrix[u][v] == 1

    def neighbors(self, v):
        row = self.matrix[v]
        return [u for u in range(self.n) if row[u]]


class ListGraph:
    """Ориентированный граф на списках смежности. Память Θ(n + m)."""

    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]

    def add_edge(self, u, v):
        self.adj[u].append(v)

    def has_edge(self, u, v):
        return v in self.adj[u]

    def neighbors(self, v):
        return self.adj[v]


# --------------------------------------------------------------- обходы
def bfs(graph, start):
    """Обход в ширину. Возвращает (порядок обхода, расстояния, предки)."""
    n = graph.n
    dist, parent, order = [-1] * n, [-1] * n, []
    dist[start] = 0
    queue = deque([start])
    while queue:
        v = queue.popleft()
        order.append(v)
        for u in graph.neighbors(v):
            if dist[u] == -1:
                dist[u] = dist[v] + 1
                parent[u] = v
                queue.append(u)
    return order, dist, parent


def dfs(graph, start):
    """Итеративный обход в глубину. Возвращает порядок посещения."""
    visited = [False] * graph.n
    order, stack = [], [start]
    while stack:
        v = stack.pop()
        if visited[v]:
            continue
        visited[v] = True
        order.append(v)
        for u in reversed(graph.neighbors(v)):
            if not visited[u]:
                stack.append(u)
    return order


# ------------------------------------------- задача варианта: поиск цикла
def find_cycle(graph):
    """Обнаружение цикла в ориентированном графе обходом в глубину с тремя цветами:
    0 — не посещена, 1 — «в обработке» (в стеке), 2 — обработана.
    Дуга в вершину цвета 1 замыкает цикл. Возвращает список вершин цикла или None.
    Итеративная реализация (явный стек итераторов): нет ограничения глубины рекурсии."""
    n = graph.n
    color, parent = [0] * n, [-1] * n
    for s in range(n):
        if color[s]:
            continue
        color[s] = 1
        stack = [(s, iter(graph.neighbors(s)))]
        while stack:
            v, it = stack[-1]
            for u in it:
                if color[u] == 0:
                    color[u] = 1
                    parent[u] = v
                    stack.append((u, iter(graph.neighbors(u))))
                    break
                if color[u] == 1:                    # обратная дуга v → u
                    cycle, x = [v], v
                    while x != u:
                        x = parent[x]
                        cycle.append(x)
                    cycle.reverse()                  # u → … → v, дуга v → u замыкает цикл
                    return cycle
            else:
                color[v] = 2
                stack.pop()
    return None


def check_cycle(graph, cycle):
    """Проверка найденного цикла: все дуги, включая замыкающую, существуют."""
    return all(graph.has_edge(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle)))


def has_cycle_kahn(graph):
    """Независимая проверка (алгоритм Кана): остались вершины — значит, есть цикл."""
    n = graph.n
    indeg = [0] * n
    for v in range(n):
        for u in graph.neighbors(v):
            indeg[u] += 1
    queue = deque(v for v in range(n) if indeg[v] == 0)
    done = 0
    while queue:
        v = queue.popleft()
        done += 1
        for u in graph.neighbors(v):
            indeg[u] -= 1
            if indeg[u] == 0:
                queue.append(u)
    return done < n


# ---------------------------------------------------------- генерация
def random_arcs(n, m, rng, acyclic=False):
    """m различных дуг (u, v), u != v. При acyclic=True все дуги направлены
    от вершины с меньшим «рангом» к большему — граф ациклический."""
    rank = list(range(n))
    rng.shuffle(rank)
    m = min(m, n * (n - 1) // 2 if acyclic else n * (n - 1))
    arcs = set()
    while len(arcs) < m:
        u, v = rng.randrange(n), rng.randrange(n)
        if u == v:
            continue
        if acyclic and rank[u] > rank[v]:
            u, v = v, u
        arcs.add((u, v))
    return sorted(arcs)


def build(cls, n, arcs):
    g = cls(n)
    for u, v in arcs:
        g.add_edge(u, v)
    return g


def deep_sizeof(obj):
    """Расход памяти списка списков, байт; каждое вложенное число учитывается один раз
    (малые числа в CPython разделяются между ячейками)."""
    total = sys.getsizeof(obj)
    unique = {}
    for row in obj:
        total += sys.getsizeof(row)
        for x in row:
            if id(x) not in unique:
                unique[id(x)] = sys.getsizeof(x)
    return total + sum(unique.values())


def timed(func, *args, repeats=3):
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), result


# ---------------------------------------------------- проверка корректности
def self_test(trials=300):
    """Сверка find_cycle с алгоритмом Кана на случайных малых графах обоих представлений."""
    rng = random.Random(9)
    with_cycle = 0
    for _ in range(trials):
        n = rng.randint(2, 25)
        m = rng.randint(0, min(3 * n, n * (n - 1)))
        arcs = random_arcs(n, m, rng, acyclic=rng.random() < 0.4)
        for cls in (MatrixGraph, ListGraph):
            g = build(cls, n, arcs)
            cyc = find_cycle(g)
            assert (cyc is not None) == has_cycle_kahn(g), (n, arcs)
            if cyc is not None:
                assert check_cycle(g, cyc), cyc
        with_cycle += cyc is not None
    return trials, with_cycle


# --------------------------------------------- И7: схема бизнес-процесса
def strongly_connected(graph):
    """Компоненты сильной связности (алгоритм Косарайю, итеративный)."""
    n = graph.n
    order, seen = [], [False] * n
    for s in range(n):
        if seen[s]:
            continue
        seen[s] = True
        stack = [(s, iter(graph.neighbors(s)))]
        while stack:
            v, it = stack[-1]
            for u in it:
                if not seen[u]:
                    seen[u] = True
                    stack.append((u, iter(graph.neighbors(u))))
                    break
            else:
                order.append(v)
                stack.pop()
    rev = ListGraph(n)
    for v in range(n):
        for u in graph.neighbors(v):
            rev.add_edge(u, v)
    comp, comps = [-1] * n, []
    for s in reversed(order):
        if comp[s] != -1:
            continue
        members, stack = [], [s]
        comp[s] = len(comps)
        while stack:
            v = stack.pop()
            members.append(v)
            for u in rev.neighbors(v):
                if comp[u] == -1:
                    comp[u] = len(comps)
                    stack.append(u)
        comps.append(sorted(members))
    return comps, rev


STEPS = ["Начало", "Приём заказа", "Проверка данных клиента", "Проверка наличия товара",
         "Запрос уточнения у клиента", "Резервирование товара", "Выставление счёта",
         "Ожидание оплаты", "Проверка оплаты", "Напоминание об оплате", "Комплектация",
         "Доставка", "Закрытие заказа", "Конец", "Отмена заказа", "Возврат средств",
         "Пересчёт скидки", "Эскалация менеджеру", "Повторная проверка менеджером",
         "Уведомление клиента", "Ожидание ответа клиента"]
ARCS = [(0, 1), (1, 2), (2, 3), (2, 4), (4, 20), (20, 4), (3, 5), (3, 14), (5, 6), (6, 7),
        (7, 8), (8, 10), (8, 9), (9, 7), (10, 11), (11, 12), (12, 13), (14, 13), (11, 19),
        (15, 12), (16, 6), (17, 18), (18, 17), (18, 13)]
START, FINAL = 0, 13


def analyze_process():
    g = build(ListGraph, len(STEPS), ARCS)
    _, dist, _ = bfs(g, START)
    reachable = [v for v in range(g.n) if dist[v] != -1]
    unreachable = [v for v in range(g.n) if dist[v] == -1]
    dead_ends = [v for v in range(g.n) if not g.neighbors(v) and v != FINAL]
    comps, rev = strongly_connected(g)
    cyclic = []
    for c in comps:
        if len(c) > 1:
            inside = set(c)
            has_exit = any(u not in inside for v in c for u in g.neighbors(v))
            cyclic.append({"steps": c, "exit": has_exit, "reachable": c[0] in reachable})
    _, back, _ = bfs(rev, FINAL)                    # из каких шагов достижим «Конец»
    cannot_finish = [v for v in reachable if back[v] == -1]
    return {"reachable": reachable, "unreachable": unreachable, "dead_ends": dead_ends,
            "cyclic": cyclic, "cannot_finish": cannot_finish}


# ---------------------------------------------------------------- графики
def make_plots(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = res["rows"]
    ns = [r["n"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in ((axes[0], "bfs", "Обход в ширину (BFS) из вершины 0"),
                           (axes[1], "cyc_dag", "Обнаружение цикла в ациклическом графе (полный обход)")):
        ax.loglog(ns, [r[key]["matrix"] for r in rows], "o-", label="матрица смежности")
        ax.loglog(ns, [r[key]["list"] for r in rows], "s-", label="списки смежности")
        ax.set_xlabel("число вершин n"); ax.set_ylabel("время, с"); ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", alpha=.3); ax.legend()
    plt.tight_layout(); plt.savefig("img/lr9_time.png", dpi=170); plt.close()

    plt.figure(figsize=(6.6, 4.2))
    x = range(len(ns))
    plt.bar([i - 0.2 for i in x], [r["mem"]["matrix"] / 2 ** 20 for r in rows], 0.4, label="матрица смежности")
    plt.bar([i + 0.2 for i in x], [r["mem"]["list"] / 2 ** 20 for r in rows], 0.4, label="списки смежности")
    plt.yscale("log"); plt.xticks(list(x), ns); plt.xlabel("число вершин n"); plt.ylabel("память, МБ")
    plt.title("Расход памяти представлениями графа"); plt.grid(True, axis="y", which="both", alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig("img/lr9_memory.png", dpi=170); plt.close()

    pr = res["prob"]
    plt.figure(figsize=(6.6, 4.2))
    plt.plot([p["c"] for p in pr], [p["share"] * 100 for p in pr], "o-")
    plt.xlabel("средняя исходящая степень m / n"); plt.ylabel("доля графов с циклом, %")
    plt.title("Вероятность цикла в случайном орграфе (n = 500)")
    plt.grid(True, alpha=.3); plt.tight_layout(); plt.savefig("img/lr9_prob.png", dpi=170); plt.close()


# ------------------------------------------------------------------- main
if __name__ == "__main__":
    res = {}
    res["selftest"] = self_test()
    print("Самопроверка: графов", res["selftest"][0], ", из них с циклом", res["selftest"][1])

    # --- 1. Малый граф: обходы и цикл
    small_arcs = [(0, 1), (1, 2), (2, 3), (3, 1), (0, 4), (4, 5), (5, 6), (7, 8), (8, 9), (6, 9)]
    small = build(ListGraph, 10, small_arcs)
    o_bfs, dist, _ = bfs(small, 0)
    res["small"] = {"arcs": small_arcs, "bfs": [v + 1 for v in o_bfs], "dist": dist,
                    "dfs": [v + 1 for v in dfs(small, 0)],
                    "cycle": [v + 1 for v in find_cycle(small)]}
    print(res["small"])

    # --- 2. Сравнение представлений, m ≈ 3n
    rng = random.Random(14)
    res["rows"] = []
    print(f"{'n':>6}{'m':>8}{'BFS матр.':>11}{'BFS спис.':>11}{'цикл DAG матр.':>16}{'цикл DAG спис.':>16}{'МБ матр.':>10}{'МБ спис.':>10}")
    for n in (500, 2000, 5000):
        m = 3 * n
        arcs = random_arcs(n, m, rng)
        dag_arcs = random_arcs(n, m, rng, acyclic=True)
        gm, gl = build(MatrixGraph, n, arcs), build(ListGraph, n, arcs)
        dm, dl = build(MatrixGraph, n, dag_arcs), build(ListGraph, n, dag_arcs)
        row = {"n": n, "m": m, "bfs": {}, "cyc_rand": {}, "cyc_dag": {}, "mem": {}}
        for key, g_m, g_l in (("bfs", gm, gl), ("cyc_rand", gm, gl), ("cyc_dag", dm, dl)):
            func = (lambda g: bfs(g, 0)) if key == "bfs" else find_cycle
            reps = 3
            tm, rm = timed(func, g_m, repeats=reps)
            tl, rl = timed(func, g_l, repeats=reps)
            row[key] = {"matrix": tm, "list": tl}
            if key == "bfs":
                assert rm[1] == rl[1]
                row["reach"] = sum(1 for d in rm[1] if d != -1)
            else:
                assert (rm is None) == (rl is None)
                if key == "cyc_rand":
                    assert rl is not None and check_cycle(gl, rl)
                    row["cycle_len"] = len(rl)
                else:
                    assert rl is None and not has_cycle_kahn(dl)
        row["mem"] = {"matrix": deep_sizeof(gm.matrix), "list": deep_sizeof(gl.adj)}
        res["rows"].append(row)
        print(f"{n:>6}{m:>8}{row['bfs']['matrix']:>11.4f}{row['bfs']['list']:>11.4f}"
              f"{row['cyc_dag']['matrix']:>16.4f}{row['cyc_dag']['list']:>16.4f}"
              f"{row['mem']['matrix'] / 2 ** 20:>10.1f}{row['mem']['list'] / 2 ** 20:>10.2f}", flush=True)
        del gm, dm

    # --- 3. Вероятность цикла в зависимости от m / n (n = 500)
    res["prob"] = []
    n, trials = 500, 200
    for c in (0.3, 0.6, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0):
        cnt = 0
        for _ in range(trials):
            g = build(ListGraph, n, random_arcs(n, int(c * n), rng))
            cyc = find_cycle(g)
            assert (cyc is not None) == has_cycle_kahn(g)
            cnt += cyc is not None
        res["prob"].append({"c": c, "share": cnt / trials})
    print(res["prob"])

    # --- 4. И7: схема бизнес-процесса
    res["process"] = {"steps": STEPS, "arcs": ARCS, "analysis": analyze_process()}
    print(res["process"]["analysis"])

    with open("res/lr9.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    make_plots(res)
