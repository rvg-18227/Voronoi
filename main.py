import argparse
from voronoi_game import VoronoiGame

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--spawn", type=int, default=5, help="Number of days after which a new unit spawns at the "
                                                             "homebase")
    parser.add_argument("--last", type=int, default=100, help="Total number of days the game goes on for")
    parser.add_argument("--seed", "-s", type=int, default=2, help="Seed used by random number generator, specify 0 to "
                                                                  "use no seed and have different random behavior on "
                                                                  "each launch")
    parser.add_argument("--port", type=int, default=8080, help="Port to start, specify -1 to auto-assign")
    parser.add_argument("--address", "-a", type=str, default="127.0.0.1", help="Address")
    parser.add_argument("--no_browser", "-nb", action="store_true", help="Disable browser launching in GUI mode")
    parser.add_argument("--no_gui", "-ng", action="store_true", help="Disable GUI")
    parser.add_argument("--log_path", default="log", help="Directory path to dump log files, filepath if "
                                                          "disable_logging is false")
    parser.add_argument("--disable_logging", action="store_true", help="Disable Logging, log_path becomes path to file")
    parser.add_argument("--disable_timeout", action="store_true", help="Disable timeouts for player code")
    parser.add_argument("--player1", "-p1", default="d", help="Specifying player 1 out of 4")
    parser.add_argument("--player2", "-p2", default="d", help="Specifying player 2 out of 4")
    parser.add_argument("--player3", "-p3", default="d", help="Specifying player 3 out of 4")
    parser.add_argument("--player4", "-p4", default="d", help="Specifying player 4 out of 4")
    parser.add_argument("--dump_state", action="store_true", help="Dump game.pkl for rendering")
    args = parser.parse_args()
    player_list = tuple([args.player1, args.player2, args.player3, args.player4])
    del args.player1
    del args.player2
    del args.player3
    del args.player4

    if args.disable_logging:
        if args.log_path == "log":
            args.log_path = "results.log"
    
    voronoi_game = VoronoiGame(player_list, args)
