const fs = require('fs');
const path = require('path');

const browsersync = require('browser-sync');
const cssnano = require('cssnano');
const glob = require('glob');
const moment = require('moment');

const gulp = require('gulp');
const debug = require('gulp-debug');

const concat = require('gulp-concat');
const dom = require('gulp-dom');
const htmlmin = require('gulp-htmlmin');
const imagemin = require('gulp-imagemin');
const mustache = require("gulp-mustache");
const plumber = require('gulp-plumber');
const postcss = require('gulp-postcss');
const rename = require('gulp-rename');
const watch = require('gulp-watch');

gulp.task('default', ['serve', 'post']);

// -------------------------------------------------------------------
// browser-sync
// -------------------------------------------------------------------

gulp.task('serve', function() {
  browsersync({
    server: './build',
    port: 4000,
    files: [
      "./build/**/*.html",
      "./build/**/*.css",
      "./build/**/img/*"
    ]
  });
});

// -------------------------------------------------------------------
// post
// -------------------------------------------------------------------

gulp.task('post',
          ['optimize-html',
           'optimize-css',
           'optimize-img',
           'generate-index']);

gulp.task('optimize-html', function() {
  var p = './src/posts/**/README.html';
  return gulp.src(p)
    .pipe(watch(p, {verbose: true}))
    .pipe(dom(function() {
      var doc = this;
      var footnotes = doc.getElementById('footnotes');
      var content = doc.getElementById('content');
      var bibliography = doc.getElementById('bibliography');
      if (bibliography)
        content.insertBefore(bibliography, footnotes);
      return doc;
    }))
    .pipe(htmlmin({
      removeComments: true,
      collapseWhitespace: true,
      removeEmptyAttributes: true,
      minifyJS: true,
      minifyCSS: true
    }))
    .pipe(rename({basename: 'index'}))
    .pipe(gulp.dest('./build/'));
});

gulp.task('optimize-css', function() {
  var processors = [
    cssnano({autoprefixer: {browsers: ['last 2 version'], add: true},
             discardComments: {removeAll: true}})];

  var p = './src/**/*.css';
  return gulp.src(p)
    .pipe(watch(p, {verbose: true}))
    .pipe(plumber())
    .pipe(postcss(processors))
    .pipe(gulp.dest('./build'));
});

gulp.task('optimize-img', function() {
  var p = './src/posts/**/img/*';
  return gulp.src(p)
    .pipe(watch(p, {verbose: true}))
    .pipe(imagemin())
    .pipe(gulp.dest('./build'));
});

gulp.task('generate-index', function() {
  var p = './src/posts';
  var articles = fs.readdirSync(p)
        .filter(function(fn) {
          return fs.statSync(path.join(p, fn)).isDirectory();
        })
        .map(function(fn) {
          var anchor = glob.sync(path.join(p, fn, '*.org'))[0];
          var mtime = fs.statSync(anchor).mtime;
          var ts = moment(mtime).format('YYYY-MM-DD ddd HH:mm');
          return {timestamp: ts,
                  href: path.join('.', fn),
                  title: fn.replace('-', ' ')};
        });

  var docs = [
    {href: 'http://gongzhitaao.org/orgcss/',
     title: 'CSS for org-exported HTML',
     img: 'http://orgmode.org/img/org-mode-unicorn-logo.png'},
    {href: 'http://gongzhitaao.org/dotemacs/',
     title: 'My living emacs',
     img: 'https://www.gnu.org/software/emacs/images/emacs.png'}];

  var view = {
    docs: docs,
    articles: articles
  };

  var p = './src/index.mustache';
  return gulp.src(p)
    .pipe(watch(p, {verbose: true}))
    .pipe(plumber())
    .pipe(mustache(view, {extension: '.html'}))
    .pipe(htmlmin({
      removeComments: true,
      collapseWhitespace: true,
      removeEmptyAttributes: true,
      minifyJS: true,
      minifyCSS: true
    }))
    .pipe(gulp.dest('./build'));
});

// -------------------------------------------------------------------
// deploy
// -------------------------------------------------------------------

gulp.task('deploy', function(){
  return gulp.src('./build/**/*')
    .pipe(gulp.dest('../gh-pages'));
});
