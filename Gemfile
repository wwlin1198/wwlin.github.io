source "https://rubygems.org"

gem "jekyll", "~> 4.3"

gem "minima", "~> 2.5", ">= 2.5.2"

group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.12"
  gem "jekyll-sitemap"
  gem "jekyll-seo-tag"
end

# For Ruby >= 3, WEBrick is no longer bundled with Ruby stdlib
# Jekyll serve depends on it, so include it explicitly
# https://github.com/jekyll/jekyll/issues/8523

gem "webrick", "~> 1.8"

# Use Windows Directory Monitor to avoid polling for changes on Windows
gem 'wdm', '>= 0.1.0' if Gem.win_platform?
