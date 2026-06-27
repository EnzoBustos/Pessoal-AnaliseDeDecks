"""Legacy compatibility entry point for archetype scraping."""

from utils.helpers import configure_utf8_console

from scraping.scrape_archetypes import main


if __name__ == "__main__":
    configure_utf8_console()
    main()
