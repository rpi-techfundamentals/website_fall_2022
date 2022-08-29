# Jupyter Class

This is a project to easily build a course website from Jupyter Book.  

Set the configuration in the config folder.
### Configure Automation File
set the configuration for datadog in `/config/datadogConfig` if your sheet will be using google sheets as a central repo
the file contains 2 variables:
1. **sheet_id**: this is the id of the file. You can find this by opening the share link URL and taking the string between `/d` and `/edit?`
2. **lastModified**: this contains a default value and is automatically updated, do not touch this.

### Using google Sheets
The book.xslx file is compatible with google sheets if you wish to create a central repo.
To do this you simple need to convert the file to a sheets file. a guide for this is avalible
[here](https://spreadsheetpoint.com/convert-excel-to-google-sheets/)

#### Limitation of google sheets
due to the nature of how sheets works, unless you are an organization drive user (i.e. your company is part of a shared workspace)
you must manually run this script once to authorize datadog to your google drive. This is part of google drives consent framework and cannot be changed.

Once this is done, your token is good for 30 days and you will not have to run it manually again until then.

You will also need to setup the [google drive API](https://developers.google.com/drive/api/v3/enable-drive-api) as this is the
only method of handling drive files. You will configure the API for a  `desktop application` and use the `OAuth2.0` method.
make sure to download the credential file as `credentials.json`. If you do not do this, the API will throw a Permission Denied error

## Building a Jupyter Book
 
Run the following command in your terminal:

```bash
jb build site/
```

If you would like to work with a clean build, you can empty the build folder by running:

```bash
jb clean site/
```

If jupyter execution is cached, this command will not delete the cached folder.

To remove the build folder (including `cached` executables), you can run:

```bash
jb clean --all site/
```

## Publishing this Jupyter Book

This repository is published automatically to `gh-pages` upon `push` to the `master` branch.

## Notes

This repository is used as a test case for [jupyter-book](https://github.com/executablebooks/jupyter-book) and
a `requirements.txt` file is provided to support this `CI` application.
