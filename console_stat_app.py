import algorythms
import time

maze_gen_algorythms = [algorythms.growing_tree, algorythms.aldous_broder, algorythms.wilson, algorythms.backtracking,
                       algorythms.binary_tree, algorythms.kruskal, algorythms.eller,
                       algorythms.modified_prim, algorythms.sidewinder, algorythms.division, algorythms.serpentine,
                       algorythms.small_rooms, algorythms.spiral]
maze_solve_algorythms = [algorythms.ai_lab_solve, algorythms.a_star, algorythms.dijkstra, algorythms.bfs]

if __name__ == '__main__':
    count = 1000
    for index in range(1, 4):
        for n in range(10, 101, 10):
            aver_t = 0.0
            for i in range(count):
                maze = maze_gen_algorythms[9](n, n)  # change num
                start = time.time()
                maze_solve_algorythms[index](maze, (0, 0), (n - 2, n - 2))
                end = time.time()
                aver_t += end - start
            aver_t /= count
            print(f"{n} {aver_t}")
        print("----------------")
