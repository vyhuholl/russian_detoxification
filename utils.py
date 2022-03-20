import progressbar

pbar = None


def show_progress(block_num: int, block_size: int, total_size: int) -> None:
    """
    Prints progress bar when downloading files with urllib.request.

    Parameters:
        block_num: int
        block_size: int
        total_size: int

    Returns:
        None
    """
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None
